#!/bin/sh

sed -i 's/"value": [0-9]*, "update": true/"value": 0, "update": true/' "$1"
