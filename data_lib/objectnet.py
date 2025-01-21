import os
from typing import Tuple

import gdown
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm
from copy import deepcopy

import data_lib

import contextlib
from datetime import datetime, timezone
import getpass
import io
import json
import pathlib
import uuid
import pickle
import hashlib
import subprocess
from os.path import join, exists

import sqlalchemy as sqla
from sqlalchemy.ext.declarative import declarative_base as sqla_declarative_base
from sqlalchemy_utils import database_exists, create_database

import concurrent.futures
import math
import random
import shutil
import time
from timeit import default_timer as timer

import boto3
import botocore
from botocore.client import Config

BASE_CLASSES = json.load(open("data_lib/00_info/OBJECTNET_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x.replace("_", " ")) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": True,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}

################### Objectnet utils sourced and modified from: https://github.com/modestyachts/imagenet-testbed/blob/master/src/mldb/ ####################

default_profile = "default"
default_cache_root_path = (pathlib.Path(__file__).parent / "../../s3_cache").resolve()

db_connection_file = (
    pathlib.Path(__file__).parent / "db_connection_string.txt"
).resolve()
DB_CONNECTION_MODE = "rds" if os.path.exists(db_connection_file) else "sqlite"
if DB_CONNECTION_MODE == "rds":
    DB_CONNECTION_STRING_RDS = open(db_connection_file, "r").readline()
DB_CONNECTION_STRING_SQLITE = (
    "sqlite:///" + str(default_cache_root_path) + "/robustness_evaluation.db"
)


def tar_directory(dir_name, target_filename):
    subprocess.run(["tar", "-cf", str(target_filename), str(dir_name)], check=True)


def untar_directory(tar_filename, target_dir, strip=None, one_top_level=False):
    cmd = ["tar", "-xf", str(tar_filename)]
    if strip:
        cmd += [f"--strip={strip}"]
    cmd += ["-C", str(target_dir)]
    if one_top_level:
        cmd += ["--one-top-level"]
    subprocess.run(cmd, check=True)
    subprocess.run(["rm", tar_filename], check=True)


def get_s3_client_vasa():
    if DB_CONNECTION_MODE == "rds":
        if default_profile in boto3.Session()._session.available_profiles:
            session = boto3.Session(profile_name=default_profile)
        else:
            session = boto3.Session()
        client = session.client(
            "s3",
            endpoint_url="https://vasa.millennium.berkeley.edu:9000",
            aws_access_key_id="robustness-eval",
            aws_secret_access_key="rtB_HizvjHVl59_HgKjOBYZJZTbXjNRHbIsBEj5D4g4",
            config=Config(connect_timeout=250, read_timeout=250),
            verify=(pathlib.Path(__file__).parent / "vasa_chain.cer").resolve(),
            region_name="us-east-1",
        )
        return client

    elif DB_CONNECTION_MODE == "sqlite":
        client = boto3.client(
            "s3",
            endpoint_url="https://vasa.millennium.berkeley.edu:9000",
            config=Config(signature_version=botocore.UNSIGNED),
            verify=(pathlib.Path(__file__).parent / "vasa_chain.cer").resolve(),
            region_name="us-east-1",
        )
        return client


def get_s3_client_google():
    client = boto3.client(
        "s3",
        endpoint_url="https://gresearch.storage.googleapis.com",
        config=Config(signature_version=botocore.UNSIGNED),
    )
    return client


# default is vasa, but some older objects are stored on google
get_s3_client = get_s3_client_google


def key_exists(bucket, key):
    # Return true if a key exists in s3 bucket
    # TODO: return None from the get functions if the key doesn't exist?
    #       (this would avoid one round-trip to S3)
    client = get_s3_client_google()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] != "404":
            raise
        return False
    except:
        raise


def get_s3_object_bytes_parallel(
    keys,
    *,
    bucket,
    cache_on_local_disk=True,
    cache_root_path=None,
    verbose=False,
    special_verbose=True,
    max_num_threads=90,
    num_tries=5,
    initial_delay=1.0,
    delay_factor=math.sqrt(2.0),
    download_callback=None,
    skip_modification_time_check=False,
):
    if cache_on_local_disk:
        assert cache_root_path is not None
        cache_root_path = pathlib.Path(cache_root_path).resolve()

        missing_keys = []
        existing_keys = []
        for key in keys:
            local_filepath = cache_root_path / key
            if not local_filepath.is_file():
                missing_keys.append(key)
                local_filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                existing_keys.append(key)

        keys_to_download = missing_keys.copy()
        if skip_modification_time_check:
            if verbose:
                print(
                    f"Skipping the file modification time check for {len(existing_keys)} keys that have local copies."
                )
            for key in existing_keys:
                if download_callback:
                    download_callback(1)
        else:
            if verbose:
                print(
                    f"Getting metadata for {len(existing_keys)} keys that have local copies ... ",
                    end="",
                )
            metadata_start = timer()
            metadata = get_s3_object_metadata_parallel(
                existing_keys,
                bucket=bucket,
                verbose=False,
                max_num_threads=max_num_threads,
                num_tries=num_tries,
                initial_delay=initial_delay,
                delay_factor=delay_factor,
                download_callback=None,
            )
            metadata_end = timer()
            if verbose:
                print(f"took {metadata_end - metadata_start:.3f} seconds")
            for key in existing_keys:
                local_filepath = cache_root_path / key
                assert local_filepath.is_file
                local_time = datetime.datetime.fromtimestamp(
                    local_filepath.stat().st_mtime, datetime.timezone.utc
                )
                remote_time = metadata[key]["LastModified"]
                if local_time <= remote_time:
                    if verbose:
                        print(f'Local copy of key "{key}" is outdated')
                    keys_to_download.append(key)
                elif download_callback:
                    download_callback(1)

        tl = threading.local()

        def cur_download_file(key):
            local_filepath = cache_root_path / key
            if verbose or special_verbose:
                print(
                    "{} not available locally or outdated, downloading from S3 ... ".format(
                        key
                    )
                )
            download_s3_file_with_backoff(
                key,
                str(local_filepath),
                bucket=bucket,
                num_tries=num_tries,
                initial_delay=initial_delay,
                delay_factor=delay_factor,
                thread_local=tl,
            )
            return local_filepath.is_file()

        if len(keys_to_download) > 0:
            download_start = timer()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_num_threads
            ) as executor:
                future_to_key = {
                    executor.submit(cur_download_file, key): key
                    for key in keys_to_download
                }
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        success = future.result()
                        assert success
                        if download_callback:
                            download_callback(1)
                    except Exception as exc:
                        print("Key {} generated an exception: {}".format(key, exc))
                        raise exc
            download_end = timer()
            if verbose:
                print(
                    "Downloading took {:.3f} seconds".format(
                        download_end - download_start
                    )
                )

        result = {}
        # TODO: parallelize this as well?
        for key in keys:
            local_filepath = cache_root_path / key
            if verbose:
                print("Reading from local file {} ... ".format(local_filepath), end="")
            with open(local_filepath, "rb") as f:
                result[key] = f.read()
            if verbose:
                print("done")
    else:
        tl = threading.local()

        def cur_get_object_bytes(key):
            if verbose:
                print("Loading {} from S3 ... ".format(key))
            return get_s3_object_bytes_with_backoff(
                key,
                bucket=bucket,
                num_tries=num_tries,
                initial_delay=initial_delay,
                delay_factor=delay_factor,
                thread_local=tl,
            )[0]

        download_start = timer()
        result = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_num_threads
        ) as executor:
            future_to_key = {
                executor.submit(cur_get_object_bytes, key): key for key in keys
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result[key] = future.result()
                    if download_callback:
                        download_callback(1)
                except Exception as exc:
                    print("Key {} generated an exception: {}".format(key, exc))
                    raise exc
        download_end = timer()
        if verbose:
            print(
                "Getting object bytes took {} seconds".format(
                    download_end - download_start
                )
            )
    return result


def get_s3_object_bytes_with_backoff(
    key,
    *,
    bucket,
    num_tries=5,
    initial_delay=1.0,
    delay_factor=math.sqrt(2.0),
    num_replicas=1,
    thread_local=None,
):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, "get_object_client"):
            thread_local.get_object_client = get_s3_client()
        client = thread_local.get_object_client
    delay = initial_delay
    num_tries_left = num_tries

    if num_replicas > 1:
        replicas_counter_len = len(str(num_replicas))
        format_string = "_replica{{:0{}d}}-{{}}".format(replicas_counter_len)
    while num_tries_left >= 1:
        try:
            if num_replicas > 1:
                cur_replica = random.randint(1, num_replicas)
                cur_key = key + format_string.format(cur_replica, num_replicas)
            else:
                cur_key = key
            read_bytes = client.get_object(Key=cur_key, Bucket=bucket)["Body"].read()
            return read_bytes, cur_key
        except:
            if num_tries_left == 1:
                raise Exception("get backoff failed " + key + " " + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def get_s3_object_metadata_with_backoff(
    key,
    *,
    bucket,
    num_tries=5,
    initial_delay=1.0,
    delay_factor=math.sqrt(2.0),
    thread_local=None,
):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, "get_object_client"):
            thread_local.get_object_client = get_s3_client()
        client = thread_local.get_object_client
    delay = initial_delay
    num_tries_left = num_tries
    while num_tries_left >= 1:
        try:
            metadata = client.head_object(Key=key, Bucket=bucket)
            return metadata
        except:
            if num_tries_left == 1:
                raise Exception("get backoff failed " + key + " " + str(delay))
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def get_s3_object_metadata_parallel(
    keys,
    bucket,
    verbose=False,
    max_num_threads=20,
    num_tries=5,
    initial_delay=1.0,
    delay_factor=math.sqrt(2.0),
    download_callback=None,
):
    tl = threading.local()

    def cur_get_object_metadata(key):
        if verbose:
            print("Loading metadata for {} from S3 ... ".format(key))
        return get_s3_object_metadata_with_backoff(
            key,
            bucket=bucket,
            num_tries=num_tries,
            initial_delay=initial_delay,
            delay_factor=delay_factor,
            thread_local=tl,
        )

    download_start = timer()
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_threads) as executor:
        future_to_key = {
            executor.submit(cur_get_object_metadata, key): key for key in keys
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result[key] = future.result()
                if download_callback:
                    download_callback(1)
            except Exception as exc:
                print("Key {} generated an exception: {}".format(key, exc))
                raise exc
    download_end = timer()
    if verbose:
        print(
            "Getting object metadata took {} seconds".format(
                download_end - download_start
            )
        )
    return result


def download_s3_file_with_caching(
    key,
    local_filename,
    *,
    bucket,
    cache_on_local_disk=True,
    cache_root_path=None,
    verbose=False,
    special_verbose=True,
    num_tries=5,
    initial_delay=1.0,
    delay_factor=math.sqrt(2.0),
    num_replicas=1,
    skip_modification_time_check=False,
):
    if cache_on_local_disk:
        assert cache_root_path is not None
        cache_root_path = pathlib.Path(cache_root_path).resolve()
        currently_cached = False

        cache_filepath = cache_root_path / key
        if not cache_filepath.is_file():
            cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            if skip_modification_time_check:
                if verbose:
                    print(
                        f"Skipping the file modification time check the local copy in the cache."
                    )
                currently_cached = True
            else:
                if verbose:
                    print(
                        f"Getting metadata to check the modification time compared to the local copy ... ",
                        end="",
                    )
                metadata_start = timer()
                metadata = get_s3_object_metadata_with_backoff(
                    key,
                    bucket=bucket,
                    num_tries=num_tries,
                    initial_delay=initial_delay,
                    delay_factor=delay_factor,
                )
                metadata_end = timer()
                if verbose:
                    print(f"took {metadata_end - metadata_start:.3f} seconds")
                local_time = datetime.datetime.fromtimestamp(
                    cache_filepath.stat().st_mtime, datetime.timezone.utc
                )
                remote_time = metadata["LastModified"]
                if local_time <= remote_time:
                    if verbose:
                        print(f'Local copy of key "{key}" is outdated')
                else:
                    currently_cached = True
        if not currently_cached:
            if verbose or special_verbose:
                print(
                    "{} not available locally or outdated, downloading from S3 ... ".format(
                        key
                    )
                )
            download_start = timer()
            download_s3_file_with_backoff(
                key,
                str(cache_filepath),
                bucket=bucket,
                initial_delay=initial_delay,
                delay_factor=delay_factor,
                num_replicas=num_replicas,
            )
            download_end = timer()
            if verbose:
                print(
                    "Downloading took {:.3f} seconds".format(
                        download_end - download_start
                    )
                )
        assert cache_filepath.is_file()
        if verbose:
            print(f"Copying to the target from the cache file {cache_filepath} ...")
        shutil.copy(cache_filepath, local_filename)
    else:
        if verbose:
            print("Loading {} from S3 ... ".format(key))
        download_start = timer()
        download_s3_file_with_backoff(
            key,
            local_filename,
            bucket=bucket,
            initial_delay=initial_delay,
            delay_factor=delay_factor,
            num_replicas=num_replicas,
        )
        download_end = timer()
        if verbose:
            print(
                "Downloading took {:.3f} seconds".format(download_end - download_start)
            )


def download_s3_file_with_backoff(
    key,
    local_filename,
    *,
    bucket,
    num_tries=5,
    initial_delay=1.0,
    delay_factor=math.sqrt(2.0),
    num_replicas=1,
    thread_local=None,
):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, "s3_client"):
            thread_local.s3_client = get_s3_client()
        client = thread_local.s3_client
    delay = initial_delay
    num_tries_left = num_tries

    if num_replicas > 1:
        replicas_counter_len = len(str(num_replicas))
        format_string = "_replica{{:0{}d}}-{{}}".format(replicas_counter_len)
    while num_tries_left >= 1:
        try:
            if num_replicas > 1:
                cur_replica = random.randint(1, num_replicas)
                cur_key = key + format_string.format(cur_replica, num_replicas)
            else:
                cur_key = key
            print(local_filename)
            client.download_file(bucket, cur_key, local_filename)
            return cur_key
        except:
            if num_tries_left == 1:
                raise Exception(
                    "download backoff failed " + " " + str(key) + " " + str(delay)
                )
            else:
                time.sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def default_option_if_needed(*, user_option, default):
    if user_option is None:
        return default
    else:
        return user_option


class S3Wrapper:
    def __init__(
        self,
        bucket,
        cache_on_local_disk=True,
        cache_root_path=default_cache_root_path,
        verbose=False,
        max_num_threads=90,
        num_tries=5,
        initial_delay=1.0,
        delay_factor=math.sqrt(2.0),
        skip_modification_time_check=False,
    ):
        self.bucket = bucket
        self.cache_on_local_disk = cache_on_local_disk
        # self.client = get_s3_client()

        if self.cache_on_local_disk:
            assert cache_root_path is not None
            self.cache_root_path = pathlib.Path(cache_root_path).resolve()
            self.cache_root_path.mkdir(parents=True, exist_ok=True)
            assert self.cache_root_path.is_dir()
        else:
            self.cache_root_path = None
        self.verbose = verbose
        self.max_num_threads = max_num_threads
        self.num_tries = num_tries
        self.initial_delay = initial_delay
        self.delay_factor = delay_factor
        self.skip_modification_time_check = skip_modification_time_check

    def download_file(
        self, key, filename, verbose=None, skip_modification_time_check=None
    ):
        cur_verbose = default_option_if_needed(
            user_option=verbose, default=self.verbose
        )
        cur_skip_time_check = default_option_if_needed(
            user_option=skip_modification_time_check,
            default=self.skip_modification_time_check,
        )
        download_s3_file_with_caching(
            key,
            filename,
            bucket=self.bucket,
            cache_on_local_disk=self.cache_on_local_disk,
            cache_root_path=self.cache_root_path,
            num_tries=self.num_tries,
            initial_delay=self.initial_delay,
            delay_factor=self.delay_factor,
            skip_modification_time_check=cur_skip_time_check,
            verbose=True,
        )

    def get(self, key, verbose=None, skip_modification_time_check=None):
        return self.get_multiple(
            [key],
            verbose=verbose,
            skip_modification_time_check=skip_modification_time_check,
        )[key]

    def get_multiple(
        self, keys, verbose=None, callback=None, skip_modification_time_check=None
    ):
        if verbose is None:
            cur_verbose = self.verbose
        else:
            cur_verbose = verbose
        cur_verbose = default_option_if_needed(
            user_option=verbose, default=self.verbose
        )
        cur_skip_time_check = default_option_if_needed(
            user_option=skip_modification_time_check,
            default=self.skip_modification_time_check,
        )
        return get_s3_object_bytes_parallel(
            keys,
            bucket=self.bucket,
            cache_on_local_disk=self.cache_on_local_disk,
            cache_root_path=self.cache_root_path,
            verbose=cur_verbose,
            max_num_threads=self.max_num_threads,
            num_tries=self.num_tries,
            initial_delay=self.initial_delay,
            delay_factor=self.delay_factor,
            download_callback=callback,
            skip_modification_time_check=cur_skip_time_check,
        )

    def exists(self, key):
        return key_exists(self.bucket, key)


class DoubleBucketS3Wrapper:
    def __init__(
        self,
        bucket_vasa,
        bucket_google,
        cache_on_local_disk=True,
        cache_root_path=default_cache_root_path,
        verbose=False,
        max_num_threads=90,
        num_tries=5,
        initial_delay=1.0,
        delay_factor=math.sqrt(2.0),
        skip_modification_time_check=False,
    ):

        self.vasa_s3wrapper = S3Wrapper(
            bucket_vasa,
            cache_on_local_disk=cache_on_local_disk,
            cache_root_path=cache_root_path,
            verbose=verbose,
            max_num_threads=max_num_threads,
            num_tries=num_tries,
            initial_delay=initial_delay,
            delay_factor=delay_factor,
            skip_modification_time_check=skip_modification_time_check,
        )

        self.google_s3wrapper = S3Wrapper(
            bucket_google,
            cache_on_local_disk=cache_on_local_disk,
            cache_root_path=cache_root_path,
            verbose=verbose,
            max_num_threads=max_num_threads,
            num_tries=num_tries,
            initial_delay=initial_delay,
            delay_factor=delay_factor,
            skip_modification_time_check=skip_modification_time_check,
        )

    def download_file(
        self, key, filename, verbose=None, skip_modification_time_check=None
    ):
        global get_s3_client
        get_s3_client = get_s3_client_vasa
        if self.vasa_s3wrapper.exists(key):
            self.vasa_s3wrapper.download_file(
                key, filename, verbose, skip_modification_time_check
            )
            return
        get_s3_client = get_s3_client_google
        if self.google_s3wrapper.exists(key):
            self.google_s3wrapper.download_file(
                key, filename, verbose, skip_modification_time_check
            )
            get_s3_client = get_s3_client_vasa
            return
        raise Exception(f"File with key {key} not found!")

    def get(self, key, verbose=None, skip_modification_time_check=None):
        global get_s3_client
        get_s3_client = get_s3_client_vasa
        if self.vasa_s3wrapper.exists(key):
            return self.vasa_s3wrapper.get(key, verbose, skip_modification_time_check)
        get_s3_client = get_s3_client_google
        if self.google_s3wrapper.exists(key):
            return_value = self.google_s3wrapper.get(
                key, verbose, skip_modification_time_check
            )
            get_s3_client = get_s3_client_vasa
            return return_value
        raise Exception(f"Data with key {key} not found!")

    def exists(self, key):
        global get_s3_client
        get_s3_client = get_s3_client_vasa
        if self.vasa_s3wrapper.exists(key):
            return True
        get_s3_client = get_s3_client_google
        if self.google_s3wrapper.exists(key):
            get_s3_client = get_s3_client_vasa
            return True
        return False


sqlalchemy_base = sqla_declarative_base()
DB_DUMP_URL = (
    "https://vasa.millennium.berkeley.edu:9000/robustness-eval/robustness_evaluation.db"
)


def download_db(default_cache_root_path):
    if not exists(join(default_cache_root_path, "robustness_evaluation.db")):
        print("downloading database dump...")
        try:
            subprocess.run(
                [
                    "wget",
                    "-P",
                    default_cache_root_path,
                    DB_DUMP_URL,
                    "--no-check-certificate",
                ],
                check=True,
            )
        except:
            gdown.download(
                "https://drive.google.com/uc?id=17XOxNxRqB_xMcHsAoewNmtuzKtdlpmzV",
                os.path.join(default_cache_root_path, "robustness_evaluation.db"),
                quiet=False,
            )


def gen_short_uuid(num_chars=None):
    num = uuid.uuid4().int
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    res = []
    while num > 0:
        num, digit = divmod(num, len(alphabet))
        res.append(alphabet[digit])
    res2 = "".join(reversed(res))
    if num_chars is None:
        return res2
    else:
        return res2[:num_chars]


def get_logdir_key(model_id):
    return "logdir/{}".format(model_id)


def get_dataset_data_key(dataset_id):
    return "datasets/{}_data.bytes".format(dataset_id)


def get_raw_input_data_key(raw_input_id):
    return "raw_inputs/{}_data.bytes".format(raw_input_id)


class Model(sqlalchemy_base):
    __tablename__ = "models"
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    extra_info = sqla.Column(sqla.JSON)
    checkpoints = sqla.orm.relationship(
        "Checkpoint",
        back_populates="model",
        cascade="all, delete, delete-orphan",
        foreign_keys="Checkpoint.model_uuid",
    )
    final_checkpoint_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("checkpoints.uuid"), nullable=True
    )
    final_checkpoint = sqla.orm.relationship(
        "Checkpoint", foreign_keys=[final_checkpoint_uuid], uselist=False
    )
    completed = sqla.Column(sqla.Boolean)
    hidden = sqla.Column(sqla.Boolean)
    logdir_filepaths = sqla.Column(sqla.JSON)

    def __repr__(self):
        return f'<Model(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class Checkpoint(sqlalchemy_base):
    __tablename__ = "checkpoints"
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    model_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("models.uuid"), nullable=False
    )
    model = sqla.orm.relationship(
        "Model", back_populates="checkpoints", foreign_keys=[model_uuid]
    )
    evaluations = sqla.orm.relationship(
        "Evaluation",
        back_populates="checkpoint",
        cascade="all, delete, delete-orphan",
        foreign_keys="Evaluation.checkpoint_uuid",
    )
    training_step = sqla.Column(sqla.BigInteger)
    epoch = sqla.Column(sqla.Float)
    username = sqla.Column(sqla.String)
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<Checkpoint(uuid="{self.uuid}", model_uuid="{self.model_uuid}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class Dataset_(sqlalchemy_base):
    __tablename__ = "datasets"
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True, nullable=False)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    size = sqla.Column(sqla.Integer)  # Number of datapoints in the dataset
    extra_info = sqla.Column(sqla.JSON)
    evaluation_settings = sqla.orm.relationship(
        "EvaluationSetting",
        back_populates="dataset",
        cascade="all, delete, delete-orphan",
        foreign_keys="EvaluationSetting.dataset_uuid",
    )
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<Dataset_(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class EvaluationSetting(sqlalchemy_base):
    __tablename__ = "evaluation_settings"
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True, nullable=False)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    dataset_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("datasets.uuid"), nullable=False
    )
    dataset = sqla.orm.relationship(
        "Dataset_", back_populates="evaluation_settings", foreign_keys=[dataset_uuid]
    )
    evaluations = sqla.orm.relationship(
        "Evaluation",
        back_populates="setting",
        cascade="all, delete, delete-orphan",
        foreign_keys="Evaluation.setting_uuid",
    )
    raw_inputs = sqla.orm.relationship(
        "RawInput",
        back_populates="setting",
        cascade="all, delete, delete-orphan",
        foreign_keys="RawInput.setting_uuid",
    )
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<EvaluationSetting(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


# For the raw float32 inputs we have for external models
class RawInput(sqlalchemy_base):
    __tablename__ = "raw_inputs"
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    setting_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("evaluation_settings.uuid"), nullable=False
    )
    setting = sqla.orm.relationship(
        "EvaluationSetting", back_populates="raw_inputs", foreign_keys=[setting_uuid]
    )
    data_shape = sqla.Column(sqla.JSON)
    data_format = sqla.Column(sqla.String)  # numpy type
    evaluations = sqla.orm.relationship(
        "Evaluation",
        back_populates="raw_input",
        cascade="all, delete, delete-orphan",
        foreign_keys="Evaluation.raw_input_uuid",
    )
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<RawInput(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class Evaluation(sqlalchemy_base):
    __tablename__ = "evaluations"
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    checkpoint_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("checkpoints.uuid"), nullable=False
    )
    checkpoint = sqla.orm.relationship(
        "Checkpoint", back_populates="evaluations", foreign_keys=[checkpoint_uuid]
    )
    # TODO: eventually make this nullable=False
    setting_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("evaluation_settings.uuid"), nullable=True
    )
    setting = sqla.orm.relationship(
        "EvaluationSetting", back_populates="evaluations", foreign_keys=[setting_uuid]
    )
    raw_input_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("raw_inputs.uuid"), nullable=True
    )
    raw_input = sqla.orm.relationship(
        "RawInput", back_populates="evaluations", foreign_keys=[raw_input_uuid]
    )
    chunks = sqla.orm.relationship(
        "EvaluationChunk",
        back_populates="evaluation",
        cascade="all, delete, delete-orphan",
        foreign_keys="EvaluationChunk.evaluation_uuid",
    )
    username = sqla.Column(sqla.String)
    extra_info = sqla.Column(sqla.JSON)
    completed = sqla.Column(sqla.Boolean)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<Evaluation(uuid="{self.uuid}", checkpoint_uuid="{self.checkpoint_uuid}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class EvaluationChunk(sqlalchemy_base):
    __tablename__ = "evaluation_chunks"
    uuid = sqla.Column(sqla.String, primary_key=True)
    creation_time = sqla.Column(
        sqla.DateTime(timezone=False), server_default=sqla.sql.func.now()
    )
    evaluation_uuid = sqla.Column(
        sqla.String, sqla.ForeignKey("evaluations.uuid"), nullable=False
    )
    evaluation = sqla.orm.relationship(
        "Evaluation", back_populates="chunks", foreign_keys=[evaluation_uuid]
    )
    username = sqla.Column(sqla.String)
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<EvaluationChunk(uuid="{self.uuid}", evaluation_uuid="{self.evaluation_uuid}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.hash() == hash(other)


class ModelRepository:
    def __init__(
        self,
        mode=DB_CONNECTION_MODE,
        sql_verbose=False,
        download_database=True,
        download_root=None,
    ):
        if download_root is not None:
            default_cache_root_path = download_root
            DB_CONNECTION_STRING_SQLITE = (
                "sqlite:///"
                + str(default_cache_root_path)
                + "/robustness_evaluation.db"
            )
        self.sql_verbose = sql_verbose
        if mode == "sqlite":
            if download_database:
                download_db(default_cache_root_path)
            self.db_connection_string = DB_CONNECTION_STRING_SQLITE
            self.engine = sqla.create_engine(
                self.db_connection_string, echo=self.sql_verbose
            )
        elif mode == "rds":
            self.db_connection_string = DB_CONNECTION_STRING_RDS
            self.engine = sqla.create_engine(
                self.db_connection_string, echo=self.sql_verbose, pool_pre_ping=True
            )
        else:
            assert False
        if not database_exists(self.engine.url):
            create_database(self.engine.url)
        self.sessionmaker = sqla.orm.sessionmaker(
            bind=self.engine, expire_on_commit=False
        )
        self.cache_root_path = default_cache_root_path
        self.s3wrapper = DoubleBucketS3Wrapper(
            bucket_vasa="robustness-eval",
            bucket_google="imagenet-testbed",
            cache_root_path=self.cache_root_path,
            verbose=False,
        )
        self.uuid_length = 10
        self.pickle_protocol = 4

    def dispose(self):
        self.engine.dispose()

    @contextlib.contextmanager
    def session_scope(self):
        session = self.sessionmaker()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def gen_short_uuid(self):
        new_id = gen_short_uuid(num_chars=self.uuid_length)
        # TODO: check that we don't have a collision with the db?
        return new_id

    def gen_checkpoint_uuid(self):
        return gen_short_uuid(num_chars=None)

    def run_query_with_optional_session(self, query, session=None):
        if session is None:
            with self.session_scope() as sess:
                return query(sess)
        else:
            return query(session)

    def run_get(self, get_fn, session=None, assert_exists=True):
        def query(sess):
            result = get_fn(sess)
            assert len(result) <= 1
            if assert_exists:
                assert len(result) == 1
            if len(result) == 0:
                return None
            else:
                return result[0]

        return self.run_query_with_optional_session(query, session)

    def get_dataset(
        self,
        *,
        uuid=None,
        name=None,
        session=None,
        assert_exists=True,
        load_evaluation_settings=False,
    ):
        if uuid is not None:
            assert type(uuid) is str
        if name is not None:
            assert type(name) is str

        def get_fn(sess):
            return self.get_datasets(
                uuids=[uuid] if uuid is not None else None,
                names=[name] if name is not None else None,
                session=sess,
                load_evaluation_settings=load_evaluation_settings,
                show_hidden=True,
            )

        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def dataset_uuid_exists(self, uuid, session=None):
        return (
            self.get_dataset(uuid=uuid, assert_exists=False, session=session)
            is not None
        )

    def get_datasets(
        self,
        *,
        uuids=None,
        names=None,
        session=None,
        load_evaluation_settings=True,
        show_hidden=False,
    ):
        cur_options = []
        if load_evaluation_settings:
            cur_options.append(sqla.orm.subqueryload(Dataset_.evaluation_settings))
        filter_list = []
        if not show_hidden:
            filter_list.append(Dataset_.hidden == False)
        if uuids is not None:
            filter_list.append(Dataset_.uuid.in_(uuids))
        if names is not None:
            filter_list.append(Dataset_.name.in_(names))

        def query(sess):
            return sess.query(Dataset_).options(cur_options).filter(*filter_list).all()

        return self.run_query_with_optional_session(query, session)

    def create_dataset(
        self,
        *,
        name,
        size,
        description=None,
        data_bytes=None,
        data_filename=None,  # use one of the two - directly uploading from a file can save memory
        extra_info=None,
        verbose=False,
    ):
        assert name is not None
        assert size is not None
        assert type(size) is int
        assert data_bytes is None or data_filename is None
        assert data_bytes is not None or data_filename is not None
        with self.session_scope() as session:
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_dataset = Dataset_(
                uuid=new_id,
                name=name,
                description=description,
                username=username,
                size=size,
                extra_info=extra_info,
                hidden=False,
            )
            key = get_dataset_data_key(new_id)
            if data_bytes is not None:
                self.s3wrapper.put(data_bytes, key, verbose=verbose)
            else:
                assert data_filename is not None
                self.s3wrapper.upload_file(data_filename, key, verbose=verbose)
            session.add(new_dataset)
        return self.get_dataset(uuid=new_id, assert_exists=True)

    def get_dataset_data(self, dataset_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.dataset_uuid_exists(dataset_uuid, session=session)
            key = get_dataset_data_key(dataset_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def download_dataset_data(self, dataset_uuid, target_filename, verbose=False):
        with self.session_scope() as session:
            assert self.dataset_uuid_exists(dataset_uuid, session=session)
            key = get_dataset_data_key(dataset_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.download_file(
                    key, target_filename, verbose=verbose
                )

    def rename_dataset(self, dataset_uuid, new_name):
        with self.session_scope() as session:
            dataset = self.get_dataset(
                uuid=dataset_uuid, session=session, assert_exists=True
            )
            old_name = dataset.name
            dataset.name = new_name
        return old_name

    def hide_dataset(self, dataset_uuid):
        with self.session_scope() as session:
            dataset = self.get_dataset(
                uuid=dataset_uuid, session=session, assert_exists=True
            )
            dataset.hidden = True


def download_dataset(dataset, file_root):
    m_repo = ModelRepository(download_root=file_root)
    dataset = m_repo.get_dataset(name=dataset)
    filedir = join(file_root, f"datasets/{dataset.name}")
    if not exists(filedir):
        filename = filedir + ".tar"
        m_repo.download_dataset_data(
            dataset_uuid=dataset.uuid, target_filename=filename
        )

        if "format-val" in dataset.name:
            strip, one_top_level = 2, False
        elif "imagenet-c" in dataset.name:
            strip, one_top_level = 6, True
        elif dataset.name in [
            "imagenetv2-matched-frequency",
            "imagenetv2-topimages",
            "imagenetv2-threshold0.7",
            "val",
        ]:
            strip, one_top_level = 3, False
        else:
            strip, one_top_level = 1, True
        untar_directory(
            filename,
            join(file_root, "datasets"),
            strip=strip,
            one_top_level=one_top_level,
        )
    close_db_connection(m_repo)
    return filedir


def close_db_connection(m_repo):
    m_repo.dispose()


######################################################################


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "OBJECTNET")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            os.makedirs(self.root, exist_ok=True)
            if not "robustness_evaluation.db" in os.listdir(self.root):
                gdown.download(
                    "https://drive.google.com/uc?id=17XOxNxRqB_xMcHsAoewNmtuzKtdlpmzV",
                    os.path.join(self.root, "robustness_evaluation.db"),
                    quiet=False,
                )

            download_dataset("objectnet-1.0-beta", os.path.join(self.root))

            assert os.path.exists(
                os.path.join(self.root, "datasets", "objectnet-1.0-beta")
            ), "Objectnet did not download correctly or is corrupted, please delete the {} directory and run the script again".format(
                self.root
            )
            # note that the count we check in the directory is 1 plus the total number of classes to account for the .DS_Store file that gets downloaded
            assert (
                len(
                    os.listdir(
                        os.path.join(self.root, "datasets", "objectnet-1.0-beta")
                    )
                )
                == 314
            ), "Objectnet did not download correctly or is corrupted, please delete the {} directory and run the script again. There should be 313 classes, but only {} downloaded.".format(
                self.root,
                len(
                    os.listdir(
                        os.path.join(self.root, "datasets", "objectnet-1.0-beta")
                    )
                ),
            )

            os.remove(os.path.join(self.root, "datasets", "dpbKHFHBwY_data.bytes"))
