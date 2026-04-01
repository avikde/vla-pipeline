# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@README.md

## Project Overview

This project primarily uses JavaScript and Python. The main codebase is a web-based MuJoCo robotics simulation with Gemini API integration. Key files include controller.js, mujoco-scene.js, and math-utils.js.

## Robotics / Control

When working with robotics IK/control code: always verify sign conventions and coordinate frame orientations before implementing. Test with a simple known case (e.g., move +X) before complex trajectories. Prefer pseudoinverse over Jacobian transpose for joint-space mapping.
