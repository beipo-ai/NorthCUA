# OSWorld / OSWorld2

Use the official benchmark's task JSON and DesktopEnv installation. Configure model
credentials in `agent.osworld.example.yaml`, then follow the
[environment guide](../../docs/environments.md). The integration owns one fresh
environment per task, drives NorthCUA through its HTTP desktop backend, saves evidence,
invokes the upstream oracle, and closes the environment.
