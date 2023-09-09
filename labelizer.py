import functools
import signal
import traceback
from PIL import Image
import requests
import sys
import os
import torch
import re
import glob
import logging
import coloredlogs
import argparse
from pprint import pformat, pprint
from tqdm import tqdm
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BlipForQuestionAnswering,
)
from photoprism.Session import Session
from photoprism.Photo import Photo

logging.basicConfig()
coloredlogs.install()
log = logging.getLogger()

# todo: mps?
device = "cuda" if torch.cuda.is_available() else "cpu"
stop_tokens = ["it", "is", "they", "yes", "no"]


@functools.cache
def is_blip2(model_name):
    return "blip2" in model_name or os.getenv("FORCE_BLIP2", False)


@functools.cache
def get_processor(model_name):
    processor_ = Blip2Processor if is_blip2(model_name) else BlipProcessor
    processor = processor_.from_pretrained(model_name)
    return processor


## ---------------------- ##


@functools.cache
def get_model(model_name, use_vqa):
    use_blip2 = is_blip2(model_name)
    log.debug(f"Loading model {model_name}, blip2 = {use_blip2}...")

    cls = (
        Blip2ForConditionalGeneration
        if use_blip2
        else (BlipForConditionalGeneration if not use_vqa else BlipForQuestionAnswering)
    )
    return cls.from_pretrained(model_name).to(device)


def generate_caption(model, processor, image, prompt):
    inputs = processor(image, prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0].strip()


def generate_caption_plain(image):
    model = get_model(os.getenv("MODEL_BLIP_HFID"), use_vqa=False)
    processor = get_processor(os.getenv("MODEL_BLIP_HFID"))

    ret = generate_caption(model, processor, image, "a photograph of")
    log.info(f"VC description: {ret}")
    return ret


def generate_caption_vqa(image):
    model = get_model(os.getenv("MODEL_VQA_HFID"), use_vqa=True)
    processor = get_processor(os.getenv("MODEL_VQA_HFID"))

    # todo: batch prompts
    ret = []
    for p in [
        "is a photo of someone taking a picture of themselves? if yes, say selfie",
        "what are the objects in this image?",
        "what is special about this image?",
        "what is the image about?",
        "list all foods in this image if there are any",
        "",
    ]:
        ret.append(generate_caption(model, processor, image, p))
    ret = list(filter(lambda x: x and x not in stop_tokens, list(set(ret))))
    log.info(f"VQA keywords: {','.join(ret)}")

    return ", ".join(ret)


def process_image(file_path, use_vqa=False):
    file = (
        requests.get(file_path, stream=True).raw
        if not os.path.exists(file_path) and file_path.startswith("http")
        else file_path
    )
    image = Image.open(file).convert("RGB")

    output = []
    keywords = generate_caption_vqa(image) if use_vqa else ""
    caption = generate_caption_plain(image)
    return keywords, caption


def process_folder(folder_path):
    # glob all image files in the folder_path recursively and yield each full path
    for file_path in glob.iglob(folder_path + "**/*", recursive=True):
        # only match image file extensions and actual files\
        if os.path.isfile(file_path) and re.match(
            r".*\.(jpg|jpeg|png|bmp|gif)", file_path, re.I
        ):
            yield file_path


def cleanup_string(s):
    # just use a csv writer
    single_quote = "'"
    double_quote = '"'
    s = re.sub("\\b(" + "|".join(stop_tokens) + ")\\b", "", s)
    s = re.sub(r",{2+}", ",", s)
    s = re.sub(r"^,|,$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace(double_quote, single_quote).strip()


def get_file_paths(root):
    # determine if root argument is a file, folder or url
    if os.path.isdir(root):
        yield from process_folder(root)
    elif os.path.isfile(root) or root.startswith("http"):
        yield root
    else:
        raise Exception(f"Invalid argument: {root}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Caption images in a folder or photoprism url"
    )

    parser.add_argument(
        "mode",
        metavar="mode",
        type=str,
        help="Operating mode - photoprism or url/folder/path; make sure to set proper environment variables for PhotoPrism",
        choices=["photoprism", "local"],
        default="local",
        nargs="?",
    )

    parser.add_argument(
        "input_path",
        metavar="input_path",
        type=str,
        nargs="?",
        help="image url or folder path/file",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output",
        type=str,
        help="output log file path (stdout)",
    )

    parser.add_argument(
        "-r",
        "--readonly",
        action="store_true",
        help="readonly mode for photoprism - no field updates will be made",
    )

    # todo:
    # photoprism filtering opts
    return parser.parse_args()


def handle_local(args, path_=None):
    outlog = open(args.output, "w+") if args.output else sys.stdout
    p = path_ or args.input_path
    for path in get_file_paths(p):
        p = os.path.abspath(path)
        try:
            ret = cleanup_string(
                ",".join(process_image(p, use_vqa=os.getenv("ENABLE_VQA", False)))
            )
            # todo: cleanup
            print(f'"{cleanup_string(p)}","{ret}"', file=outlog)
            return ret
        except Exception as e:
            print(traceback.format_exc())
            log.error(f"{p},{e}")
    return None


def get_file_extension(filename):
    return filename.split(".")[-1].lower()


def delete_local(path):
    if not os.getenv("PHOTOPRISM_DOWNLOAD_NO_DELETE", False):
        log.info(f"Removing downloaded file {path}")
        os.unlink(path)


def handle_photoprism_photo(photo, photo_instance, readonly=True):
    if not photo or not photo.get("Hash") or not photo.get("Type") == "image":
        # todo: proper logging
        log.debug(f"Skipping photo - {pformat(photo)}")
        return None

    hash = photo["Hash"]
    file_extension = get_file_extension(photo["FileName"])

    # log.debug(pprint(photo))

    p = os.path.abspath(
        os.path.join(os.getenv("PHOTOPRISM_DOWNLOAD_PATH"), f"{hash}.{file_extension}")
    )

    log.info(f"Downloading {hash} from PhotoPrism to {p}")

    if photo_instance.download_file(
        hash=hash, path=os.getenv("PHOTOPRISM_DOWNLOAD_PATH"), filename=hash
    ) and os.path.exists(p):
        (keywords, caption) = process_image(p, use_vqa=os.getenv("ENABLE_VQA", False))

        if not readonly:
            photo_instance.update_photo_description_and_keywords(
                photo["UID"],
                caption
                + (
                    f" ({keywords})" if keywords else ""
                ),  # PP keyword updates seem broken?
                keywords,
            )
    else:
        log.error(f"Failed to download {pformat(photo)}")
    return True


def handle_photoprism(args):
    pp_session = Session(
        os.getenv("PHOTOPRISM_USERNAME"),
        os.getenv("PHOTOPRISM_PASSWORD"),
        os.getenv("PHOTOPRISM_BASE_DOMAIN"),
        use_https=os.getenv("PHOTOPRISM_USE_HTTPS", True),
        verify_cert=False if os.getenv("PHOTOPRISM_ALLOW_SELF_SIGNED", False) else True,
    )
    pp_session.create()

    log.info(f"Connected to PhotoPrism: {pp_session.session_id}")
    # todo: pagination

    photo_instance = Photo(pp_session)
    num_photos = int(os.getenv("PHOTOPRISM_BATCH_SIZE", 10))
    offset = 0

    data = photo_instance.search(
        query="original:*", count=num_photos, offset=offset, order="newest"
    )

    while data:
        log.info(
            f"Fetched {len(data)} photos from PhotoPrism (offset={offset}, pagesize={num_photos})..."
        )

        for photo in data:
            offset += 1
            handle_photoprism_photo(photo, photo_instance, readonly=args.readonly)

        data = photo_instance.search(
            query="original:*", count=num_photos, offset=offset
        )


# sigint trap
def maybe_trap_sigint():
    def trap_sigint(signum, frame):
        log.info("Exiting...")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, trap_sigint)
        signal.signal(signal.SIGTERM, trap_sigint)
    except:
        pass


# preloads models and does a basic sanity check
def validate_env(args=None):
    if not (os.getenv("MODEL_VQA_HFID", False) and os.getenv("MODEL_BLIP_HFID", False)):
        raise Exception(
            "Please set one or both of MODEL_VQA_HFID and MODEL_BLIP_HFID environment variables"
        )

    if not (os.getenv("ENABLE_VQA", False) or os.getenv("ENABLE_CAPTION", False)):
        raise Exception(
            "Please set one or both of ENABLE_VQA and ENABLE_CAPTION environment variables"
        )

    if os.getenv("ENABLE_VQA"):
        if not os.getenv("MODEL_VQA_HFID"):
            raise Exception(
                "Please set MODEL_VQA_HFID environment variable when ENABLE_VQA is set"
            )
        get_model("MODEL_VQA_HFID", True)

    if os.getenv("ENABLE_CAPTION"):
        if not os.getenv("MODEL_BLIP_HFID"):
            raise Exception(
                "Please set MODEL_BLIP_HFID environment variable when ENABLE_CAPTION is set"
            )
        get_model("MODEL_BLIP_HFID", False)

    if args and args.mode == "photoprism":
        for field in ["PASSWORD", "USERNAME", "BASE_DOMAIN"]:
            if not os.getenv(f"PHOTOPRISM_{field}"):
                raise Exception(
                    f"Please set PHOTOPRISM_{field} environment variable when operating in photoprism mode"
                )


def main():
    maybe_trap_sigint()

    args = get_args()
    validate_env(args)

    if args.mode == "local":
        handle_local(args)
    elif args.mode == "photoprism":
        handle_photoprism(args)
    else:
        args.print_help()


if __name__ == "__main__":
    main()
