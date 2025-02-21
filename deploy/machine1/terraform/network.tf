#############################
# VPC
#############################
resource "aws_vpc" "test_env" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "test-env"
  }
}

#############################
# Subnet
#############################
resource "aws_subnet" "subnet_uno" {
  vpc_id            = aws_vpc.test_env.id
  cidr_block        = cidrsubnet(aws_vpc.test_env.cidr_block, 3, 1)
  availability_zone = "us-west-2a"

  tags = {
    Name = "test-env-subnet-uno"
  }
}

#############################
# Internet Gateway
#############################
resource "aws_internet_gateway" "test_env_gw" {
  vpc_id = aws_vpc.test_env.id

  tags = {
    Name = "test-env-gw"
  }
}

#############################
# Route Table & Association
#############################
resource "aws_route_table" "route_table_test_env" {
  vpc_id = aws_vpc.test_env.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.test_env_gw.id
  }

  tags = {
    Name = "test-env-route-table"
  }
}

resource "aws_route_table_association" "subnet_association" {
  subnet_id      = aws_subnet.subnet_uno.id
  route_table_id = aws_route_table.route_table_test_env.id
}

#############################
# Security Group
#############################
resource "aws_security_group" "ingress_all_test" {
  name   = "allow-all-sg"
  vpc_id = aws_vpc.test_env.id

  # SSH inbound (port 22)
  ingress {
    description = "SSH inbound"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] 
  }

  # HTTP inbound (port 80)
  ingress {
    description = "HTTP inbound"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS inbound (port 443)
  ingress {
    description = "HTTPS inbound"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # (Optional) If your Flask dev server runs on port 5000:
 ingress {
     description = "Flask dev server inbound"
     from_port   = 5000
     to_port     = 5000
     protocol    = "tcp"
     cidr_blocks = ["0.0.0.0/0"]
  }

  # Egress for all traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "allow-http-https-ssh"
  }
}