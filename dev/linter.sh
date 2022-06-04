#!/bin/bash -e

flake8_version="4.0.1"
isort_version="5.10.1"
yapf_version="0.32.0"

# cd to project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

{
  isort --version | grep $isort_version > /dev/null
} || {
  echo "Linter requires 'isort==$isort_version' !"
  exit 1
}

{
  yapf --version | grep $yapf_version > /dev/null
} || {
  echo "Linter requires 'yapf==$yapf_version' !"
  exit 1
}

{
  flake8 --version | grep $flake8_version > /dev/null
} || {
  echo "Linter requires 'flake8==$flake8_version' !"
  exit 1
}

echo "Running isort ..."
isort .

echo "Running yapf ..."
yapf --recursive --in-place .

echo "Running flake8 ..."
flake8 .
