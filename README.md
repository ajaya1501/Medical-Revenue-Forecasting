#run the below code in ec2 instance on ubuntu 

sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv
mkdir mlflow
cd mlflow
pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell


## Then set aws credentials
aws configure

mlflow server -h 0.0.0.0 --default-artifact-root s3://mllllfflow-buc-11
#open Public IPv4 DNS to the port 5000

#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-13-203-206-18.ap-south-1.compute.amazonaws.com:5000/