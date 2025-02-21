##################################################
# Input Variables
##################################################

# A friendly tag/name for your EC2 instance.
variable "ami_name" {
  type        = string
  description = "Name (tag) for the AWS instance"
  default     = "my-rl-instance"
}

# The AMI ID to use (e.g., Amazon Linux 2 or Ubuntu in us-west-2).
variable "ami_id" {
  type        = string
  description = "AMI ID for the instance"
  default     = "ami-00c257e12d6828491"
}

# The EC2 instance type to launch (e.g., t2.micro, t3.small, etc.).
variable "instance_type" {
  type        = string
  description = "AWS instance type"
  #default = "g4dn.2xlarge"
  #default     = "t2.medium"
  default     = "c6i.8xlarge"
}

# The **existing** AWS key pair name (as created in the AWS Console),
# e.g. "nmysore_aws_1"
variable "aws_key_pair_name" {
  type        = string
  description = "Name of the existing AWS key pair to use"
  default     = "nmysore_aws_1"
}

# The **local path** to the private key (.pem) file you downloaded from AWS.
# This is only used in the SSH output instructions, so you remember which file
# to use when you connect. For example: "./nmysore_aws_1.pem"
variable "local_private_key_path" {
  type        = string
  description = "Local path to the downloaded .pem file for SSH"
  default     = ""
}