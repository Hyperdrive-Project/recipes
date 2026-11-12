#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# Stops only this project's server container (nhd-server).
${DOCKER:-sudo docker} rm -f nhd-server >/dev/null 2>&1 && echo "stopped nhd-server" || echo "no nhd-server"
