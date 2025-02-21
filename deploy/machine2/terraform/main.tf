#############################
# Terraform + Provider Setup
#############################
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  # Region in which your key pair and instance should reside
  region = "us-west-2"
}

#############################
# EC2 Instance
#############################
resource "aws_instance" "app_server" {
  ami               = var.ami_id
  instance_type     = var.instance_type
  key_name          = var.aws_key_pair_name
  subnet_id         = aws_subnet.subnet_uno.id
  vpc_security_group_ids = [aws_security_group.ingress_all_test.id]

  # Increase root volume size
  root_block_device {
    volume_size = 50  # Increase this value (in GB)
    volume_type = "gp3"  # Use 'gp3' for better performance or 'gp2' (default)
    iops        = 3000
    throughput  = 125
    delete_on_termination = true
  }

  tags = {
    Name = var.ami_name
  }
}

#############################
# Attach an Elastic IP (Optional)
#############################
resource "aws_eip" "ip_test_env" {
  instance = aws_instance.app_server.id
  vpc      = true
}

#############################
# Output SSH Instructions
#############################
output "ssh_instructions" {
  description = "How to SSH into the instance"
  value = <<EOT
To connect (assuming Amazon Linux 2):
  ssh -i ${var.local_private_key_path} ec2-user@${aws_eip.ip_test_env.public_ip}

If it's Ubuntu:
  ssh -i ${var.local_private_key_path} ubuntu@${aws_eip.ip_test_env.public_ip}

Adjust the username to match your AMI, and make sure your .pem file is chmod 400.
EOT
}