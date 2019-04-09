#!/usr/bin/env bash

# These are the steps needed to set-up a remote server, e.g. a Ubuntu distribution on GCP

# Git setup
USER_MAIL=TODO
USER_NAME=TODO
git config --global user.email ${USER_MAIL}
git config --global user.name ${USER_NAME}

# Add key to your GitHub profile (unless repo is public at the time of reading)
ssh-keygen
cat .ssh\id_pub
git clone git@github.com:nbckr/fact-check.git
cd fact-check

# Install some required tools
sudo apt-get update &&
sudo apt-get install python3-pip &&
sudo apt-get install unzip &&
sudo apt-get install jq

# Install dependencies
pip3 install -r requirements.txt
