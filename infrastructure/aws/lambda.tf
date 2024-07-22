module "lambda-cleanup" {
  source                = "../../../i-ai-core-infrastructure//modules/lambda"
  file_path             = data.archive_file.code.output_path
  function_name         = "${local.name}-cleanup-lambda"
  handler               = "lambda.lambda_handler"
  runtime               = "python3.12"
  source_code_hash      = data.archive_file.code.output_base64sha256
  iam_role_name         = "${local.name}-redbox-cleanup-lambda-role"
  layers                = [aws_lambda_layer_version.layer.arn]
  timeout               = 60
  environment_variables = {
    FILE_EXPIRY_IN_SECONDS : (local.django_app_environment_variables.FILE_EXPIRY_IN_DAYS * 24 * 60 * 60),
    POSTGRES_HOST : module.rds.db_instance_address,
    POSTGRES_DB : module.rds.db_instance_name,
    POSTGRES_USER : module.rds.rds_instance_username,
    POSTGRES_PASSWORD : module.rds.rds_instance_db_password,
    ELASTIC__CLOUD_ID : var.cloud_id,
    ELASTIC__API_KEY : var.elastic_api_key,
    BUCKET_NAME : "i-dot-ai-${var.env}-redbox",
    ELASTIC_ROOT_INDEX : "redbox-data-${var.env}",
    SLACK_NOTIFICATION_URL: var.slack_url
  }
  aws_security_group_ids = [aws_security_group.service_security_group.id]
  subnet_ids             = data.terraform_remote_state.vpc.outputs.private_subnets
}

resource "aws_security_group" "service_security_group" {
  vpc_id      = data.terraform_remote_state.vpc.outputs.vpc_id
  description = "${local.name} cleanup lambda security group"
  name        = "${local.name}-cleanup-lambda-sg"
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group_rule" "lambda_to_rds_egress" {
  type                     = "egress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = module.rds.postgres_sg_id
  security_group_id        = aws_security_group.service_security_group.id
  description              = "Allow requests from the lambda to get to the RDS"
}

resource "aws_security_group_rule" "lambda_to_443_egress" {
  type              = "egress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  ipv6_cidr_blocks  = ["::/0"]
  security_group_id = aws_security_group.service_security_group.id
}

data "archive_file" "code" {
  type        = "zip"
  source_dir  = "${path.module}/cleanup_lambda/code"
  output_path = "${path.module}/cleanup_lambda/code.zip"
}


resource "aws_lambda_layer_version" "layer" {
  layer_name          = "cleanup-layer"
  filename            = data.archive_file.layer.output_path
  source_code_hash    = data.archive_file.layer.output_base64sha256
  compatible_runtimes = ["python3.12"]
}


data "archive_file" "layer" {
  type        = "zip"
  source_dir  = "${path.module}/cleanup_lambda/layer"
  output_path = "${path.module}/cleanup_lambda/layer.zip"
  depends_on  = [null_resource.pip_install]
}


resource "null_resource" "pip_install" {
  triggers = {
    shell_hash = sha256(file("${path.module}/cleanup_lambda/requirements.txt"))
  }

  provisioner "local-exec" {
    command = "python3 -m pip install -r ./cleanup_lambda/requirements.txt -t ${path.module}/cleanup_lambda/layer/python"
  }
}


# ---------------------------


module "lambda-embeddings" {
  source                = "../../../i-ai-core-infrastructure//modules/lambda"
  file_path             = data.archive_file.code.output_path
  function_name         = "${local.name}-regen-embeddings-lambda"
  handler               = "lambda.lambda_handler"
  runtime               = "python3.12"
  source_code_hash      = data.archive_file.code.output_base64sha256
  iam_role_name         = "${local.name}-redbox-embeddings-lambda-role"
  layers                = [aws_lambda_layer_version.embeddings_layer.arn]
  timeout               = 60
  environment_variables = {
    FILE_EXPIRY_IN_SECONDS : (local.django_app_environment_variables.FILE_EXPIRY_IN_DAYS * 24 * 60 * 60),
    POSTGRES_HOST : module.rds.db_instance_address,
    POSTGRES_DB : module.rds.db_instance_name,
    POSTGRES_USER : module.rds.rds_instance_username,
    POSTGRES_PASSWORD : module.rds.rds_instance_db_password,
    ELASTIC__CLOUD_ID : var.cloud_id,
    ELASTIC__API_KEY : var.elastic_api_key,
    BUCKET_NAME : "i-dot-ai-${var.env}-redbox",
    ELASTIC_ROOT_INDEX : "redbox-data-${var.env}",
    SLACK_NOTIFICATION_URL: var.slack_url
  }
  aws_security_group_ids = [aws_security_group.embeddings_security_group.id]
  subnet_ids             = data.terraform_remote_state.vpc.outputs.private_subnets
}

resource "aws_security_group" "embeddings_security_group" {
  vpc_id      = data.terraform_remote_state.vpc.outputs.vpc_id
  description = "${local.name} embeddings lambda security group"
  name        = "${local.name}-embeddings-lambda-sg"
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group_rule" "embeddings_lambda_to_rds_egress" {
  type                     = "egress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = module.rds.postgres_sg_id
  security_group_id        = aws_security_group.embeddings_security_group.id
  description              = "Allow requests from the embeddings lambda to get to the RDS"
}

resource "aws_security_group_rule" "embeddings_lambda_to_443_egress" {
  type              = "egress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  ipv6_cidr_blocks  = ["::/0"]
  security_group_id = aws_security_group.embeddings_security_group.id
}

data "archive_file" "embeddings_code" {
  type        = "zip"
  source_dir  = "${path.module}/regen_embeddings_lambda/code"
  output_path = "${path.module}/regen_embeddings_lambda/code.zip"
}


resource "aws_lambda_layer_version" "embeddings_layer" {
  layer_name          = "embeddings-layer"
  filename            = data.archive_file.embeddings_layer.output_path
  source_code_hash    = data.archive_file.embeddings_layer.output_base64sha256
  compatible_runtimes = ["python3.12"]
}


data "archive_file" "embeddings_layer" {
  type        = "zip"
  source_dir  = "${path.module}/regen_embeddings_lambda/layer"
  output_path = "${path.module}/regen_embeddings_lambda/layer.zip"
  depends_on  = [null_resource.embeddings_pip_install]
}


resource "null_resource" "embeddings_pip_install" {
  triggers = {
    shell_hash = sha256(file("${path.module}/regen_embeddings_lambda/requirements.txt"))
  }

  provisioner "local-exec" {
    command = "python3 -m pip install -r ./regen_embeddings_lambda/requirements.txt -t ${path.module}/regen_embeddings_lambda/layer/python"
  }
}
