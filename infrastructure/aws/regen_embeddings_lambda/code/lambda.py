import json
import logging
import os

import boto3
import pg8000.dbapi
import requests
from botocore.exceptions import ClientError
from elasticsearch import ApiError, Elasticsearch
from pg8000.native import literal
from requests.exceptions import RequestException

logger = logging.getLogger()
logger.setLevel("INFO")


def delete_from_elastic_and_s3(core_id, name, elastic_client, elastic_index, s3):
    try:
        # delete chunks
        elastic_client.delete_by_query(
            index=elastic_index,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": [
                                        {"term": {"parent_file_uuid.keyword": str(core_id)}},
                                        {"term": {"metadata.parent_file_uuid.keyword": str(core_id)}},
                                    ]
                                }
                            },
                        ]
                    }
                }
            },
        )

        # delete file from elastic
        elastic_client.delete(index=elastic_index, id=str(core_id))

    except ClientError:
        logger.exception("Error deleting File %s from Elastic", name)
        return False

    else:
        # delete from S3
        return bool(delete_from_s3(name, s3))


def delete_from_s3(name, s3):
    try:
        s3.delete_object(
            Bucket=os.environ["BUCKET_NAME"],
            Key=str(name),
        )

    except ApiError:
        logger.exception("Error deleting File %s from S3", name)
        return False

    else:
        return True


def delete_files(files):
    elastic_client = Elasticsearch(cloud_id=os.environ["ELASTIC__CLOUD_ID"], api_key=os.environ["ELASTIC__API_KEY"])
    elastic_index = f'{os.environ["ELASTIC_ROOT_INDEX"]}-file'
    s3 = boto3.client("s3")
    results = {
        "success": [],
        "failure": [],
    }

    for django_id, core_id, name in files:
        logger.info("Deleting file uuid:  %s", core_id)

        if elastic_client.exists(index=elastic_index, id=str(core_id)):
            deletion_success = delete_from_elastic_and_s3(core_id, name, elastic_client, elastic_index, s3)

            if deletion_success:
                results["success"].append(django_id)
            else:
                results["failure"].append(django_id)

        else:
            logger.info("file uuid not found:  %s", core_id)
            delete_from_s3(name, s3)
            results["failure"].append(django_id)

    return results


def post_summary_to_slack(message):
    try:
        r = requests.post(
            os.environ["SLACK_NOTIFICATION_URL"],
            data=json.dumps({"text": message}),
            timeout=60,
            headers={"Content-Type": "application/json"},
        )

        r.raise_for_status()

    except RequestException:
        logger.exception("Error trying to communicate with Slack")


def lambda_handler(event, context):  # noqa: ARG001 unused args
    try:
        file_expiry_in_seconds = os.environ["FILE_EXPIRY_IN_SECONDS"]

        # Connect to pg database to find relevant files
        db_params = {
            "host": os.environ["POSTGRES_HOST"],
            "database": os.environ["POSTGRES_DB"],
            "user": os.environ["POSTGRES_USER"],
            "password": os.environ["POSTGRES_PASSWORD"],
        }
        try:
            connection = pg8000.dbapi.Connection(**db_params)
            cursor = connection.cursor()

            cursor.execute(
                # using parameters doesn't work for this statement;
                # literal() used to reduce risk of SQL injection in f-string
                f"""SELECT id, core_file_uuid, original_file
                    FROM redbox_core_file
                    WHERE redbox_core_file.last_referenced < (NOW() - INTERVAL '{
                        literal(int(file_expiry_in_seconds))
                    } seconds')
                    AND redbox_core_file.status NOT IN ('deleted', 'errored')
                """  # noqa: S608 uses literal() as mitigation
            )

            results = delete_files(cursor.fetchall())

            if results["success"]:
                logger.info(tuple(literal(i)[1:-1] for i in results["success"]))
                if len(results["success"]) == 1:
                    cursor.execute(
                        f"""UPDATE redbox_core_file SET status = 'deleted'
                            WHERE id = {literal(results["success"][0])}
                        """  # noqa: S608 - uses literal() as mitigation
                    )
                else:
                    cursor.execute(
                        f"""UPDATE redbox_core_file SET status = 'deleted'
                            WHERE id IN {tuple(literal(i)[1:-1] for i in results["success"])}
                        """  # noqa: S608 - uses literal() as mitigation
                    )

            if results["failure"]:
                logger.info(tuple(literal(i)[1:-1] for i in results["failure"]))
                if len(results["failure"]) == 1:
                    cursor.execute(
                        f"""UPDATE redbox_core_file SET status = 'errored'
                            WHERE id = {literal(results["failure"][0])}
                        """  # noqa: S608 - uses literal() as mitigation
                    )
                else:
                    cursor.execute(
                        f"""UPDATE redbox_core_file SET status = 'errored'
                            WHERE id IN {tuple(literal(i)[1:-1] for i in results["failure"])}
                        """  # noqa: S608 - uses literal() as mitigation
                    )

            connection.commit()
            connection.close()
            logger.info("Database connection closed.")

            post_summary_to_slack(
                f"""File deletion summary :put_litter_in_its_place:
                    :tada: These files were successfully deleted {results["success"]}
                    :do_not_litter: These files were not able to be deleted {results["failure"]}
                """
            )

        except pg8000.dbapi.DatabaseError:
            logger.exception("Error connecting to the postgres database")
            post_summary_to_slack(
                """File deletion task failed to run :in-progress:
                """
            )

        return {"statusCode": 200, "body": json.dumps("Task has run successfully")}

    except Exception as exception:
        logger.exception("Exception occurred")
        post_summary_to_slack(
            """File deletion task failed to run :in-progress:
            """
        )
        return {"message": f"General exception {exception} occurred. Exiting..."}
