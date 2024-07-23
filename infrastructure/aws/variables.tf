variable "account_id" {
  type        = string
  description = "AWS Account ID"
}

variable "cloud_id" {
  type        = string
  description = "Cloud ID for elastic cloud instance"
}

variable "cognito_usernames" {
  type        = list(string)
  description = "List of usernames to be added"
  default     = []
}

variable "contact_email" {
  description = "The email to use for 'contact us' pages"
  type        = string
}

variable "developer_ips" {
  type        = list(string)
  description = "List of developer IPs"
}

variable "django_secret_key" {
  type        = string
  description = "cryptographic signature for django app"
}

variable "domain_name" {
  type        = string
  description = "The base domain name for the project"
}

variable "ecr_repository_uri" {
  type        = string
  description = "ECR repo uri"
}

variable "elastic_api_key" {
  type        = string
  description = "API Key for elastic cloud instance"
}

variable "env" {
  type        = string
  description = "Environment"
}

variable "external_ips" {
  type        = list(string)
  description = "List of external IPs"
}

variable "from_email" {
  type        = string
  sensitive   = true
  description = "GOV notify from email"
}

variable "govuk_notify_plain_email_template_id" {
  type        = string
  sensitive   = true
  description = "GOV notify plain template id"
}

variable "govuk_notify_api_key" {
  type        = string
  sensitive   = true
  description = "GOV notify API key"
}

variable "hosted_zone_id" {
  type        = string
  description = "Route 53 Hosted Zone"
}

variable "image_tag" {
  type        = string
  description = "Image tag"
}

variable "internal_ips" {
  type        = list(string)
  description = "IP's of No10 and CO"
}

variable "azure_openai_model" {
  type        = string
  default     = "gpt-3.5-turbo"
  description = "OPENAI model to use"
}

variable "openai_api_key" {
  type        = string
  sensitive   = true
  default     = null
  description = "OPENAI api key"
}

variable "openai_api_version" {
  type        = string
  default     = "2023-12-01-preview"
  description = "OPENAI API version"
}

variable "azure_openai_api_key" {
  type        = string
  sensitive   = true
  default     = null
  description = "The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource."
}

variable "azure_openai_endpoint" {
  type        = string
  default     = null
  description = "The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource."
}

variable "project_name" {
  type        = string
  description = "Name of project"
}

variable "region" {
  type        = string
  default     = "eu-west-2"
  description = "AWS region"
}

variable "rules" {
  description = "The rules to add to this WAF"
  type        = list(object({}))
  default     = []
}

variable "sentry_dsn" {
  description = "The sentry dsn to send sentry logs to"
  type        = string
  default     = null
}

variable "sentry_environment" {
  description = "The sentry environment to send sentry logs to"
  type        = string
  default     = null
}

variable "scope" {
  description = "Scope of the WAF, either 'CLOUDFRONT' or 'REGIONAL'"
  type        = string
  default     = "REGIONAL"
}

variable "slack_url" {
  description = "The slack webhook url"
  type        = string
}

variable "state_bucket" {
  type        = string
  description = "Name of the S3 bucket to use a terraform state"
}

variable "universal_tags" {
  type        = map(string)
  description = "Map to tag resources with"
}

variable "use_case" {
  description = "Use case/resource for WAF"
  type        = string
  default     = "load_balancer"
}

variable "team_name" {
  type        = string
  description = "The name of the team"
}

variable "publicly_accessible" {
  type        = bool
  description = "Flag to determine if the database is publicly accessible"
}

variable "context_window_size" {
  type        = number
  default     = 8000
  description = "The size of the AI's context window"
}

variable "rag_k" {
  type        = number
  default     = 30
  description = "how many chunks to retrieve to form the basis of the response"
}

variable "rag_desired_chunk_size" {
  type        = number
  default     = 300
  description = "how large should the chunk sizes be"
}

variable "rag_num_candidates" {
  type    = number
  default = 10
}

variable "elbow_filter_enabled" {
  type        = bool
  default     = true
  description = "whether the elbow filter should be enabled"
}

variable "chat_system_prompt" {
  type        = string
  default     = <<EOT
You are an AI assistant called Redbox tasked with answering questions and providing information objectively.
    EOT
  description = "base chat prompt"
}

variable "chat_question_prompt" {
  type        = string
  default     = "{question}\n=========\n Response: "
  description = "how to construct chat with no documents"
}

variable "chat_with_docs_system_prompt" {
  type        = string
  default     = <<EOT
You are an AI assistant called Redbox tasked with answering questions on user provided documents and providing information objectively.
    EOT
  description = "base chat with documents prompt"
}

variable "chat_with_docs_question_prompt" {
  type        = string
  default     = "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "
  description = "how to construct chat with documents"
}

variable "chat_with_docs_reduce_system_prompt" {
  type        = string
  default     = <<EOT
You are an AI assistant tasked with answering questions on user provided documents.
Your goal is to answer the user question based on list of summaries in a coherent manner
Please follow these guidelines while answering the question: \n
1) Identify and highlight key points,\n
2) Avoid repetition,\n
3) Ensure the answer is easy to understand,\n
4) Maintain the original context and meaning.\n
    EOT
  description = "base chat large documents reduce prompt"
}

variable "chat_with_docs_reduce_question_prompt" {
  type        = string
  default     = "Question: {question}. \n\n Documents: \n\n {summaries} \n\n Answer: "
  description = "how to construct chat with documents reduce step"
}

variable "retrieval_system_prompt" {
  type        = string
  default     = <<EOT
Given the following conversation and extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If a user asks for a particular format to be returned, such as bullet points, then please use that format.
If a user asks for bullet points you MUST give bullet points.
If the user asks for a specific number or range of bullet points you MUST give that number of bullet points.
Use **bold** to highlight the most question relevant parts in your response.
If dealing dealing with lots of data return it in markdown table format.
    EOT
  description = "base retrieval prompt"
}

variable "retrieval_question_prompt" {
  type        = string
  default     = "{question} \n=========\n{formatted_documents}\n=========\nFINAL ANSWER: "
  description = "how to construct retrieval"
}

variable "condense_system_prompt" {
  type        = string
  default     = <<EOT
Given the following conversation and a follow up question, generate a follow up question to be a standalone question. You are only allowed to generate one question in response. Include sources from the chat history in the standalone question created, when they are available. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    EOT
  description = "Base condense question prompt"
}

variable "condense_question_prompt" {
  type        = string
  default     = "{question}\n=========\n Standalone question: "
  description = "How to construct a condensed question"
}

variable "summarisation_system_prompt" {
  type        = string
  default     = <<EOT
You are an AI assistant tasked with summarizing documents.
Your goal is to extract the most important information and present it in
a concise and coherent manner. Please follow these guidelines while summarizing:
1) Identify and highlight key points,
2) Avoid repetition,
3) Ensure the summary is easy to understand,
4) Maintain the original context and meaning.
    EOT
  description = "base summarisation prompt"
}

variable "summarisation_question_prompt" {
  type        = string
  default     = "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "
  description = "how to construct summarization"
}


variable "llm_max_tokens" {
  type        = number
  default     = 1024
  description = "LLM token length"
}

variable "embedding_document_field_name" {
  type        = string
  default     = "azure_embedding"
  description = "embedding document field name"
}

variable "embedding_max_retries" {
  type        = number
  default     = 10
  description = "Number of retries to external embedding services (rate limiting)"
}

variable "embedding_retry_min_seconds" {
  type        = number
  default     = 5
  description = "Number of seconds to wait before retry to external embedding services (rate limiting)"
}

variable "embedding_retry_max_seconds" {
  type        = number
  default     = 120
  description = "Maximum number of seconds to wait before retry to external embedding services (rate limiting)"
}


variable "summarisation_chunk_max_tokens" {
  type        = number
  default     = 20000
  description = "Maximum size (in tokens) of chunk used in summarisation"
}

variable "django_command" {
  type        = string
  default     = "delete_expired_data"
  description = "Name of Django management to be run. Use with caution"
}
