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
from .photoprism_client.photoprism.Session import Session
from .photoprism_client.photoprism.Photo import Photo

logging.basicConfig()
coloredlogs.install()
log = logging.getLogger()

use_vqa = os.getenv("USE_VQA", False)
use_blip2 = os.getenv("USE_BLIP2", False) and not use_vqa

# e.g. salesforce/blip
model_name = os.getenv("MODEL_HUGGING_FACE_ID")

# todo: mps?
device = "cuda" if torch.cuda.is_available() else "mps"


@functools.cache
def get_processor():
    processor = None

    def _get():
        # todo: document processor choice
        processor_ = Blip2Processor if use_blip2 else BlipProcessor
        processor = processor_.from_pretrained(model_name)
        return processor

    return processor or _get()


## ---------------------- ##


@functools.cache
def get_model():
    log.info(f"Loading model {model_name}, blip2 = {use_blip2}...")

    cls = (
        Blip2ForConditionalGeneration
        if use_blip2
        else (BlipForConditionalGeneration if not use_vqa else BlipForQuestionAnswering)
    )
    return cls.from_pretrained(model_name).to(device)


def generate_caption(image, prompt):
    model = get_model()
    proc = get_processor()
    inputs = proc(image, prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_text = proc.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    return generated_text


def generate_caption_plain(image):
    return generate_caption(image, "a photograph of")


def generate_caption_vqa(image):
    ret = []
    for p in [
        "what are the objects in this image?",
        "what is special about this image?",
        "what is the image about?",
        "",
    ]:
        ret.append(generate_caption(image, p))
    ret = list(set(ret))
    return ", ".join(ret)


def process_image(file_path):
    file = (
        requests.get(file_path, stream=True).raw
        if not os.path.exists(file_path) and file_path.startswith("http")
        else file_path
    )
    image = Image.open(file).convert("RGB")

    if os.getenv("USE_VQA"):
        return generate_caption_vqa(image)
    return generate_caption_plain(image)


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
    stop_tokens = [
        "it",
        "is",
        "they",
    ]

    s = re.sub("\\b(" + "|".join(stop_tokens) + ")\\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace(double_quote, single_quote).strip()


def get_file_paths(root):
    # determine if root argument is a file, folder or url
    if os.path.isfile(root) or root.startswith("http"):
        yield root
    elif os.path.isdir(root):
        yield from process_folder(root)
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
    )
    parser.add_argument(
        "-i",
        "--input-path",
        metavar="path",
        type=str,
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
            ret = cleanup_string(process_image(p))
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
    num_photos = 500
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
            if not photo or not photo.get("Hash") or not photo.get("Type") == "image":
                # todo: proper logging
                log.debug(f"Skipping photo - {pformat(photo)}")
                continue

            hash = photo["Hash"]
            file_extension = get_file_extension(photo["FileName"])

            # log.debug(pprint(photo))

            p = os.path.abspath(
                os.path.join(
                    os.getenv("PHOTOPRISM_DOWNLOAD_PATH"), f"{hash}.{file_extension}"
                )
            )

            log.info(f"Downloading {hash} from PhotoPrism to {p}")

            if photo_instance.download_file(
                hash=hash, path=os.getenv("PHOTOPRISM_DOWNLOAD_PATH"), filename=hash
            ) and os.path.exists(p):
                caption = handle_local(args, p)
                if caption:
                    log.info(f"Generated description for {photo['UID']}: {caption}")

                    if not args.readonly:
                        photo_instance.update_photo_description(photo["UID"], caption)
                delete_local(p)
            else:
                log.error(f"Failed to download {pformat(photo)}")

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


def main():
    maybe_trap_sigint()

    args = get_args()
    if args.mode == "local":
        handle_local(args)
    elif args.mode == "photoprism":
        handle_photoprism(args)


if __name__ == "__main__":
    main()
