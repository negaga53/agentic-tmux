#!/usr/bin/env python3
"""
Simple W2 agent that listens for 'round-start:<n>' messages from Arbitrator,
chooses rock/paper/scissors, sends 'pick:<choice>' back, and prints round results
and game-over messages. Uses the MCP tools via the agentic.worker_mcp API.

This script runs a polling loop using receive_message(timeout=60).
"""

import os
import random
import time

from agentic.worker_mcp import send_to_agent, receive_message, list_agents


CHOICES = ["rock", "paper", "scissors"]


def choose_move():
    # simple random strategy
    return random.choice(CHOICES)


def main():
    # Identify arbitrator agent id by listing agents (assumes role contains 'Arbitrator' or id 'Arbitrator')
    agents_info = list_agents()
    arbitrator_id = None
    for a in agents_info.get("agents", []):
        if a.get("role", "").lower().find("arbitrator") != -1 or a.get("id") == "Arbitrator":
            arbitrator_id = a.get("id")
            break
    # Fallback to W1 if not found
    if not arbitrator_id:
        arbitrator_id = "W1"

    print(f"W2: using arbitrator id: {arbitrator_id}")

    running = True
    while running:
        msg = receive_message(timeout=60)
        if msg.get("status") == "no_message":
            # nothing received within timeout; continue polling
            continue

        if msg.get("status") == "received":
            text = msg.get("message", "")
            sender = msg.get("from")
            print(f"Received from {sender}: {text}")

            if text.startswith("round-start:"):
                # choose and send pick
                move = choose_move()
                send_to_agent(agent_id=arbitrator_id, message=f"pick:{move}")
                print(f"Sent pick:{move} to {arbitrator_id}")

            elif text.startswith("round-result"):
                print(f"Round result: {text}")

            elif text.startswith("game-over"):
                print(f"Game over: {text}")
                running = False

            else:
                # Echo other messages to console
                print(f"Other message: {text}")

    print("W2: exiting")


if __name__ == "__main__":
    main()
