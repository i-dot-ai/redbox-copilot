
data "aws_iam_policy_document" "ecs_exec_role_policy" {
  # checkov:skip=CKV_AWS_111:Allow for write access without constraints
  # checkov:skip=CKV_AWS_356:Allow for policies to not have resource limits
  statement {
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListBucket",
      "s3:DeleteObject",
      "s3:GetObjectVersion",
      "s3:ListMultipartUploadParts",
    ]
    resources = [
      "arn:aws:s3:::${local.bucket_name}",
      "arn:aws:s3:::${local.bucket_name}/*"
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "elasticache:*"
    ]
    resources = [module.elasticache.redis_arn]
  }
}

resource "aws_iam_policy" "redbox_policy" {
  name        = "backend-${var.project_name}-${terraform.workspace}-policy"
  description = "Allow application instance to log"
  policy      = data.aws_iam_policy_document.ecs_exec_role_policy.json
}

resource "aws_iam_role_policy_attachment" "redbox_role_policy" {
  role       = module.core-api.ecs_task_execution_exec_role_name
  policy_arn = aws_iam_policy.redbox_policy.arn
}
