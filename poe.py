import poe

#client = poe.Client("XKQqnn1N3CnByXnxZB0AhA%3D%3D",proxy="socks5://127.0.0.1:7890")
client = poe.Client("XKQqnn1N3CnByXnxZB0AhA%3D%3D")

message = """
```
我会在 <> 中给出一些高级命令，你需要理解 <> 中的命令，然后将该命令分解成一个或多个有序的子任务 (Task) 。这些子任务会修改相应的状态 state ，你需要保证连续的两个子任务 (Task) 之间的状态是匹配的。你可以输出你思考的结果，但最后应该输出子任务的有序列表 (TaskList) 。接下来我将描述各个子任务、完成这些子任务将导致的状态以及输出的格式。 

子任务 (Task) 以及对应的状态如下：
- `drawer-open`: 打开桌子上的抽屉 (drawer) ，该任务将抽屉 (drawer) 的状态 (state) 从 "CLOSED" 变为 "OPENED" ;
- `drawer-close`: 关闭桌子上的抽屉 (drawer) ，该任务将抽屉 (drawer) 的状态 (state) 从 "OPENED" 变为 "CLOSED" ;
- `coffee-push`: 将咖啡杯 (cup) 放到咖啡机 (machine) 的位置，该任务将咖啡杯 (cup) 的状态 (state) 从 "AIR" 变为 "MACHINE" ;
- `coffee-button`: 按下咖啡机 (machine) 的开关，该任务能够 "接一杯咖啡"，该任务需要确保咖啡杯 (cup) 在咖啡机 (machine) 所在的位置才能执行，也就是咖啡杯 (cup) 的状态 (state) 必须是 "MACHINE"，该任务不会改变任何状态;
- `coffee-pull`: 把咖啡杯 (cup) 从咖啡机 (machine) 的位置取走，这意味着咖啡杯 (cup) 必须在咖啡机 (machine) 的位置才能被取走，也就是咖啡杯 (cup) 的状态 (state) 必须是 "MACHINE"，该任务将咖啡杯的状态从 "MACHINE" 变为 "AIR" ；
- `shelf-place`: 把咖啡杯 (cup) 放到架子 (shelf) 上，该任务将咖啡杯 (cup) 的状态从 "AIR" 变为 "SHELF" ;
- `shelf-pickup`: 把咖啡杯 (cup) 从架子 (shelf) 上取走，该任务将咖啡杯 (cup) 的状态从 "SHELF" 变为 "AIR" ;
- `drawer-place`: 把咖啡杯 (cup) 放到抽屉 (drawer) 里，这意味着咖啡杯 (cup) 的状态必须是 "AIR" ，抽屉 (drawer) 的状态必须是 "OPENED" ，该任务将咖啡杯 (cup) 的状态从 "AIR" 变为 "DRAWER" ;
- `drawer-pickup`: 把咖啡杯 (cup) 从抽屉里 (drawer) 取走，这意味着咖啡杯 (cup) 的状态必须是 "DRAWER"，抽屉 (drawer) 的状态必须是 "OPENED" ，该任务将咖啡杯 (cup) 的状态从 "DRAWER" 变为 "AIR" ;

最后部分的格式应该为json格式：{TaskList: ["task-1", "task-2", ...]}，其中，每一个元素都是以上定义的可执行的子任务。如果用户在<>中给出的指令你无法完成，最后的输出应该为：{TaskList: [None]}。

下面是一些例子：

输入：<请把咖啡杯放到抽屉里，现在物体的状态如下：{drawer: CLOSED, cup: SHELF}>
输出：
Step 1. 打开抽屉，让抽屉的状态 (state) 从 "CLOSED" 变成 "OPENED" ，对应的子任务为 `drawer-open` ；
Step 2. 将咖啡杯 (cup) 从架子上 (shelf) 取走，让咖啡杯 (cup) 的状态 (state) 从 "SHELF" 变成 "AIR" ，对应的子任务为 `shelf-pickup` ；
Step 3. 把咖啡杯 (cup) 放到抽屉 (drawer) 里，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "DRAWER" ，对应的子任务为 `drawer-place` ；
Step 4. 关上抽屉 (drawer)，让抽屉 (drawer) 的状态 (state) 从 "OPENED" 变成 "CLOSED" ，对应的子任务为 `drawer-close` ；
Step 5. {TaskList: [drawer-open,shelf-pickup,drawer-place,drawer-close]}

输入: <倒一杯咖啡，现在物体状态如下：{drawer: CLOSED, cup: DRAWER}>
输出: 
Step 1. 打开抽屉，让抽屉的状态 (state) 从 "CLOSED" 变成 "OPENED" ，对应的子任务为 `drawer-open` ；
Step 2. 将咖啡杯 (cup) 从抽屉里 (drawer) 取走，让咖啡杯 (cup) 的状态 (state) 从 "DRAWER" 变成 "AIR" ，对应的子任务为 `drawer-pickup` ；
Step 3. 把咖啡杯 (cup) 放到咖啡机处 (machine) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "MACHINE" ，对应的子任务为 `coffee-push` ；
Step 4. 为了接一杯咖啡，应该按下咖啡机 (machine) 的按钮，对应的子任务为 `coffee-button` ；
Step 5. {TaskList: [drawer-open, drawer-pickup, coffee-push, coffee-button]}

输入：<把咖啡杯放到架子处，现在物体的状态如下：{drawer: CLOSED, cup: DRAWER}>
输出：
Step 1. 打开抽屉，让抽屉的状态 (state) 从 "CLOSED" 变成 "OPENED" ，对应的子任务为 `drawer-open` ；
Step 2. 将咖啡杯 (cup) 从抽屉 (drawer) 取走，让咖啡杯 (cup) 的状态 (state) 从 "DRAWER" 变成 "AIR" ，对应的子任务为 `drawer-pickup` ；
Step 3. 把咖啡杯 (cup) 放到架子上 (shelf) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "SHELF" ，对应的子任务为 `shelf-place` ；
Step 4. {TaskList: [drawer-open,drawer-pickup,shelf-place]}

输入：<把咖啡杯放到架子处，现在物体的状态如下：{cup: MACHINE}>
输出：
Step 1. 咖啡杯 (cup) 现在在咖啡机 (machine) 处，即，状态 (state) 为 "MACHINE"，应该先取出咖啡杯，让咖啡杯 (cup) 的状态 (state) 从 "MACHINE" 变成 "AIR" ，对应的子任务为 `coffee-pull` ；
Step 2. 把咖啡杯 (cup) 放到架子上 (shelf) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "SHELF" ，对应的子任务为 `shelf-place` ；
Step 3. {TaskList: [coffee-pull,shelf-place]}

输入：<把咖啡杯放到抽屉里，当前各个物体状态如下：{cup: MACHINE, drawer: CLOSED}>
输出: 
Step 1. 打开抽屉，让抽屉的状态 (state) 从 "CLOSED" 变成 "OPENED" ，对应的子任务为 `drawer-open` ；
Step 2. 咖啡杯 (cup) 现在在咖啡机 (machine) 处，即，状态 (state) 为 "MACHINE"，应该先取出咖啡杯，让咖啡杯 (cup) 的状态 (state) 从 "MACHINE" 变成 "AIR" ，对应的子任务为 `coffee-pull` ；
Step 3. 把咖啡杯 (cup) 放到抽屉 (drawer) 里，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "DRAWER" ，对应的子任务为 `drawer-place` ；
Step 4. 关上抽屉 (drawer)，让抽屉 (drawer) 的状态 (state) 从 "OPENED" 变成 "CLOSED" ，对应的子任务为 `drawer-close` ；
Step 5. {TaskList: [drawer-open,coffee-pull,drawer-place,drawer-close]}

输入：<接一杯咖啡，然后把咖啡放到架子上，当前各个物体状态如下：{cup: DRAWER, drawer: CLOSED}>
输出: 
Step 1. 打开抽屉，让抽屉的状态 (state) 从 "CLOSED" 变成 "OPENED" ，对应的子任务为 `drawer-open` ；
Step 2. 将咖啡杯 (cup) 从抽屉里 (drawer) 取走，让咖啡杯 (cup) 的状态 (state) 从 "DRAWER" 变成 "AIR" ，对应的子任务为 `drawer-pickup` ；
Step 3. 把咖啡杯 (cup) 放到咖啡机处 (machine) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "MACHINE" ，对应的子任务为 `coffee-push` ；
Step 4. 为了接一杯咖啡，应该按下咖啡机 (machine) 的按钮，对应的子任务为 `coffee-button` ；
Step 5. 把咖啡杯 (cup) 从咖啡机 (machine) 取出，让咖啡杯 (cup) 的状态 (state) 从 "MACHINE" 变成 "AIR" ，对应的子任务为 `coffee-pull` ；
Step 6. 把咖啡杯 (cup) 放到架子上 (shelf) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "SHELF" ，对应的子任务为 `shelf-place` ；
Step 7. {TaskList: [drawer-open,drawer-pickup,coffee-push,coffee-button,coffee-pull,shelf-place]}

输入：<拿到杯子，接一杯咖啡，然后把咖啡放到架子上，当前各个物体状态如下：{cup: SHELF}>
输出: 
Step 1. 将咖啡杯 (cup) 从架子上 (shelf) 取走，让咖啡杯 (cup) 的状态 (state) 从 "SHELF" 变成 "AIR" ，对应的子任务为 `shelf-pickup` ；
Step 2. 把咖啡杯 (cup) 放到咖啡机处 (machine) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "MACHINE" ，对应的子任务为 `coffee-push` ；
Step 3. 为了接一杯咖啡，应该按下咖啡机 (machine) 的按钮，对应的子任务为 `coffee-button` ；
Step 4. 把咖啡杯 (cup) 从咖啡机 (machine) 取出，让咖啡杯 (cup) 的状态 (state) 从 "MACHINE" 变成 "AIR" ，对应的子任务为 `coffee-pull` ；
Step 5. 把咖啡杯 (cup) 放到架子上 (shelf) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "SHELF" ，对应的子任务为 `shelf-place` ；
Step 6. {TaskList: [shelf-pickup,coffee-push,coffee-button,coffee-pull,shelf-place]}

输入：<拿到杯子，接一杯咖啡，然后把咖啡放到架子上，当前各个物体状态如下：{cup: DRAWER, drawer: OPENED}>
输出: 
Step 1. 把咖啡杯 (cup) 从抽屉里 (drawer) 取走，让咖啡杯 (cup) 的状态 (state) 从 "DRAWER" 变成 "AIR" ，对应的子任务为 `drawer-pickup` ；
Step 2. 把咖啡杯 (cup) 放到咖啡机处 (machine) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "MACHINE" ，对应的子任务为 `coffee-push` ；
Step 3. 为了接一杯咖啡，应该按下咖啡机 (machine) 的按钮，对应的子任务为 `coffee-button` ；
Step 4. 把咖啡杯 (cup) 从咖啡机 (machine) 取出，让咖啡杯 (cup) 的状态 (state) 从 "MACHINE" 变成 "AIR" ，对应的子任务为 `coffee-pull` ；
Step 5. 把咖啡杯 (cup) 放到架子上 (shelf) ，让咖啡杯 (cup) 的状态 (state) 从 "AIR" 变成 "SHELF" ，对应的子任务为 `shelf-place` ；
Step 6. {TaskList: [drawer-pickup,coffee-push,coffee-button,coffee-pull,shelf-place]}
```

你应该能够理解其他类似的指令，在后续我在<>中给出指令后，你应该给出你的推理过程，最终给出指定格式的子任务序列。如果理解了，请完成以下任务：
<拿到杯子，接一杯咖啡，然后把咖啡放到架子上，当前各个物体状态如下：{cup: DRAWER,drawer: CLOSED}>
"""

#发送message
for chunk in client.send_message("a2", message):
  print(chunk)
print(chunk["text"])

#清除message记录
client.purge_conversation("a2", count=2)


'''
#查看历史数据
message_history = client.get_message_history("a2", count=10)
print(f"Last {len(message_history)} messages:")
for message in message_history:
  node = message["node"]
  print(f'{node["author"]}: {node["text"]}')
'''
