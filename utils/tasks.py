import os

TASKS = ["bikeMotorbike", "catDog", "manWoman"]

taskIndex = os.getenv('TASK', 0)

if taskIndex.isdigit() and int(taskIndex) > 0 and int(taskIndex) < len(TASKS):
  taskIndex = int(taskIndex)
else:
  print("[⛔ TASK NOT VALID] Default one will be used")

  taskIndex = 0

currentTask = TASKS[taskIndex]