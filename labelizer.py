from datetime import datetime
import functools
import json
from random import shuffle
import signal
import sqlite3
import traceback
from typing import List
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
from urllib import parse as urlparse
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BlipForQuestionAnswering,
)


logging.basicConfig()
coloredlogs.install()
log = logging.getLogger()


def use_captions():
    return os.getenv("ENABLE_CAPTION", "false") == "true"


def use_vqa():
    return os.getenv("ENABLE_VQA", "false") == "true"


@functools.cache
def get_vqa_prompts():
    with open(os.getenv("VQA_PROMPTS_FILE")) as f:
        return json.load(f)


# todo: mps?
supported_image_extensions = ["jpg", "jpeg", "png", "bmp", "gif"]
device = "cuda" if torch.cuda.is_available() else "cpu"
stop_tokens = [
    "it",
    "is",
    "they",
    "yes",
    "no",
    "none",
    "don't know",
    "unknown",
    "i'm not sure",
    "i don't know",
    "no animals",
    "not food",
    "not a selfie",
    "no words",
    "joke",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "not dog",
    "not animals",
    "no one",
    "in ",
]


# sqlite wrapper to retrieve a value by the provided key
def get_last_update(photo_id):
    db_path = os.getenv("STATE_DB_PATH", None)
    if not db_path:
        return None
    try:
        with sqlite3.connect(db_path) as db:
            cursor = db.cursor()
            cursor.execute(
                "SELECT last_update FROM last_update WHERE photo_id=?", (photo_id,)
            )
            ret = cursor.fetchone()
            cursor.close()

            return ret
    except Exception:
        return None


def store_last_update(photo_id, last_update, photo_data=""):
    db_path = os.getenv("STATE_DB_PATH", None)
    if not db_path:
        return None
    try:
        with sqlite3.connect(db_path) as db:
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO last_update VALUES (?, ?, ?)",
                (photo_id, last_update, photo_data),
            )
            db.commit()
    except Exception:
        log.warn(traceback.format_exc())


def create_state_db():
    db_path = os.getenv("STATE_DB_PATH", None)
    if not db_path:
        return None
    try:
        with sqlite3.connect(db_path) as db:
            cursor = db.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS last_update (photo_id text, last_update text, full_photo_data text)"
            )

            return True
    except Exception:
        log.warn(traceback.format_exc())
        return False


@functools.cache
def is_blip2(model_name):
    return "blip2" in model_name or os.getenv("FORCE_BLIP2", False)


@functools.cache
def get_processor(model_name):
    processor_ = Blip2Processor if is_blip2(model_name) else BlipProcessor
    processor = processor_.from_pretrained(model_name)
    return processor


## ---------------------- ##


# @functools.cache
__model_instances = {}


def get_model(model_name):
    global __model_instances
    if model_name in __model_instances:
        return __model_instances[model_name]

    use_blip2 = is_blip2(model_name)
    vqa = use_vqa() and model_name == os.getenv("MODEL_VQA_HFID")
    log.info(f"Loading model {model_name}, blip2 = {use_blip2}, vqa = {vqa}...")

    if use_blip2:
        cls = Blip2ForConditionalGeneration
    else:
        cls = BlipForQuestionAnswering if vqa else BlipForConditionalGeneration

    __model_instances[model_name] = cls.from_pretrained(model_name).to(device)
    return __model_instances[model_name]


def is_iterable(o):
    if isinstance(o, str):
        return False

    try:
        iter(o)
        return True
    except Exception:
        return False


def cleanup_list(l):
    ret = []

    if not is_iterable(l):
        return cleanup_list(
            [
                l,
            ]
        )

    for i, el in enumerate(l):
        if is_iterable(el):
            ret[i] = cleanup_list(el)
        else:
            el = el.lower()
            if el in stop_tokens:
                continue
            ret.append(el)

    return list(set(filter(lambda el: el, ret)))


def generate_caption_multi(
    model, processor, image, prompt: List[str] | str | List[List[str]]
):
    new_prompt = []
    relabel = []
    if not is_iterable(prompt):
        new_prompt.append(prompt)
    else:
        for x in prompt:
            if is_iterable(x):
                new_prompt.append(x[0])
                relabel.append(x[1])
            else:
                new_prompt.append(x)
                relabel.append(None)
    if not new_prompt:
        return ""
    ret = []
    for i, p in enumerate(new_prompt):
        caption = generate_caption(model, processor, image, p)
        if caption[0] == "yes" and relabel[i]:
            ret.append(relabel[i])
        else:
            ret.append(caption[0])
    return ret


def generate_caption(
    model, processor, image, prompt: List[str] | str | List[List[str]]
):
    if is_iterable(prompt):
        generated_text = generate_caption_multi(model, processor, image, prompt)
    else:
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    log.info(f"Model prompt: {prompt} -> {generated_text}")
    return (
        generated_text
        if is_iterable(generated_text)
        else [
            generated_text,
        ]
    )


def generate_caption_plain(image):
    if not use_captions():
        return ""

    model = get_model(os.getenv("MODEL_BLIP_HFID"))
    processor = get_processor(os.getenv("MODEL_BLIP_HFID"))

    ret = generate_caption(model, processor, image, "a photograph of")
    log.info(f"VC description: {ret}")
    return ret[0]


def generate_caption_vqa(image, prompts):
    if use_vqa():
        model = get_model(os.getenv("MODEL_VQA_HFID"))
        processor = get_processor(os.getenv("MODEL_VQA_HFID"))
    else:
        model = get_model(os.getenv("MODEL_BLIP_HFID"))
        processor = get_processor(os.getenv("MODEL_BLIP_HFID"))
    # todo: batch prompts
    p = prompts or get_vqa_prompts()
    ret = []

    ret = cleanup_list(generate_caption(model, processor, image, p))
    log.info(f"VQA keywords: {ret}")

    return ",".join(ret)


def process_image(file_path: str | bytes, prompts=None):
    if is_url_of_image_file(file_path) and not os.path.exists(file_path):
        image = Image.open(requests.get(file_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(file_path).convert("RGB")

    keywords = generate_caption_vqa(image, prompts) if use_vqa() else ""
    caption = generate_caption_plain(image) if use_captions() else ""
    return keywords, caption


def is_image_filename(filename):
    pat = r".*\.(" + "|".join(supported_image_extensions) + ")"
    return re.match(pat, filename, re.I)


def process_folder(folder_path):
    return filter(
        lambda file_path: os.path.isfile(file_path)
        and is_image_filename(os.path.basename(file_path)),
        glob.iglob(folder_path + "**/*", recursive=True),
    )


def cleanup_string(s):
    # just use a csv writer
    single_quote = "'"
    double_quote = '"'
    s = re.sub("\\b(" + "|".join(stop_tokens) + ")\\b", "", s)
    s = re.sub(r",{2+}", ",", s)
    s = re.sub(r"^,|,$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace(double_quote, single_quote).strip()


def is_url_of_image_file(url: str):
    # make it's a URI
    if not url or not isinstance(url, str) or not url.strip().startswith("http"):
        return False

    parsed_url = urlparse.urlparse(url)
    return is_image_filename(parsed_url.path)


def get_file_paths(root_):
    # resolve any symbolic links
    root = os.path.realpath(root_) if not is_iterable(root_) else root_

    # determine if root argument is a file, folder or url
    if is_iterable(root):
        for r in root:
            yield from get_file_paths(r)
    elif os.path.exists(root):
        if os.path.isdir(root):
            yield from process_folder(root)
        elif os.path.isfile(root) or is_url_of_image_file(root):
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
        nargs="*",
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

    parser.add_argument(
        "-q",
        "--query",
        metavar="query",
        default="original:*",
        type=str,
        help="photoprism query (default: original:*)",
    )

    parser.add_argument(
        "-m",
        "--max_items",
        metavar="max_items",
        default=0,
        type=int,
        help="max items to process",
    )

    parser.add_argument(
        "-ppof",
        "--offset",
        metavar="offset",
        default=0,
        type=int,
        help="starting offset for photoprism",
    )

    parser.add_argument(
        "-orderby",
        "--order_by",
        metavar="order_by",
        default="newest",
        type=str,
        help="photoprism ordering mode: oldest, newest, etc",
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
            ret = cleanup_string(",".join(process_image(p)))
            # todo: cleanup
            print(f'"{cleanup_string(p)}","{ret}"', file=outlog)
        except Exception as e:
            print(traceback.format_exc())
            log.error(f"{p},{e}")


def get_file_extension(filename):
    return filename.split(".")[-1].lower()


def delete_local(path):
    if "true" != os.getenv("PHOTOPRISM_DOWNLOAD_NO_DELETE", "false").lower():
        log.info(f"Removing downloaded file {path}")
        os.unlink(path)


def handle_photoprism_photo(photo, photo_instance, readonly=True):
    if not photo or not photo.get("Hash") or not photo.get("Type") == "image":
        # todo: proper logging
        log.debug(f"Skipping photo - {pformat(photo)}")
        return None

    hash = photo["Hash"]
    file_extension = get_file_extension(photo["FileName"])
    if file_extension not in supported_image_extensions:
        log.debug(f"Skipping photo - {pformat(photo)}")
        return None

    # log.debug(pprint(photo))

    p = os.path.abspath(
        os.path.join(os.getenv("PHOTOPRISM_DOWNLOAD_PATH"), f"{hash}.{file_extension}")
    )

    log.info(f"Fetching {hash}/{photo['UID']} -> {p}")

    if photo_instance.download_file(
        hash=hash, path=os.getenv("PHOTOPRISM_DOWNLOAD_PATH"), filename=hash
    ) and os.path.exists(p):
        (keywords, caption) = process_image(p)

        if not readonly:
            photo_instance.update_photo_description_and_keywords(
                photo["UID"],
                caption
                + (
                    f" ({keywords})" if keywords else ""
                ),  # PP keyword updates seem broken?
                keywords,
            )
        delete_local(p)
    else:
        log.error(f"Failed to download {pformat(photo)}")
    return True


def handle_photoprism(args):
    # imports here to relieve need for requirements.txt / local mode etc
    from photoprism.Session import Session
    from photoprism.Photo import Photo

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
    offset = args.offset or 0
    order = args.order_by

    def do_search():
        return photo_instance.search(
            query=args.query, count=num_photos, offset=offset, order=order
        )

    data = do_search()
    have_db = not args.readonly and create_state_db()

    while data:
        log.info(
            f"Fetched {len(data)} photos from PhotoPrism (offset={offset}, pagesize={num_photos})..."
        )

        if os.getenv("PHOTOPRISM_SHUFFLE", False):
            shuffle(data)

        for photo in data:
            offset += 1

            if have_db:
                last_update = get_last_update(photo["UID"])
                if last_update:
                    log.info(
                        f"Skipping {photo['UID']}, already processed on {last_update}"
                    )
                    continue

            handle_photoprism_photo(photo, photo_instance, readonly=args.readonly)

            # store current time as last update
            if have_db:
                store_last_update(
                    photo["UID"], datetime.now().isoformat(), json.dumps(photo)
                )

            if args.max_items and offset >= args.max_items:
                log.info(f"Done - processed {offset} items")
                return

        data = do_search()


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

    if not (os.getenv("ENABLE_VQA", False) and os.getenv("ENABLE_CAPTION", False)):
        raise Exception(
            "Please set one or both of ENABLE_VQA and ENABLE_CAPTION environment variables"
        )

    if use_vqa():
        if not (
            os.getenv("MODEL_VQA_HFID", None)
            and os.getenv("VQA_PROMPTS_FILE", None)
            and os.path.exists(os.getenv("VQA_PROMPTS_FILE"))
        ):
            raise Exception(
                "Please set MODEL_VQA_HFID environment variable when ENABLE_VQA is set"
            )
        get_model(os.getenv("MODEL_VQA_HFID"))

    if use_captions():
        if not os.getenv("MODEL_BLIP_HFID"):
            raise Exception(
                "Please set MODEL_BLIP_HFID environment variable when ENABLE_CAPTION is set"
            )
        get_model(os.getenv("MODEL_BLIP_HFID"))

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
