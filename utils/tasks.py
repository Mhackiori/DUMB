import os

TASKS = ["catDog", "manWoman", "bikeMotorbike"]

taskIndex = os.getenv('TASK', 0)

if taskIndex.isdigit() and int(taskIndex) > 0 and int(taskIndex) < len(TASKS):
  taskIndex = int(taskIndex)
else:
  print("[⛔ TASK NON VALIDO] Verrà usato quello predefinito")

  taskIndex = 0

currentTask = TASKS[taskIndex]