export aws_region='us-east-2'
export aws_account_id='380955945980'
aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com

docker tag vision-ezout_vision $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/ezout-dev:vision

docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/ezout-dev:vision