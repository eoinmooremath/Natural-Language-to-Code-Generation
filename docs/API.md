# Robot API Documentation

This document describes the robot API that the Robot-LoRA system generates code for. The API is designed to be intuitive and covers all common robot operations.

## 🤖 Core Robot API

### Movement Commands

#### `robot.move_forward(distance, speed)`
Move the robot forward by a specified distance.

**Parameters:**
- `distance` (float): Distance to move in meters (0.1 - 10.0)
- `speed` (float): Movement speed in m/s (0.1 - 2.0)

**Examples:**
```python
robot.move_forward(distance=3.0, speed=1.0)
robot.move_forward(distance=0.5, speed=0.3)  # Slow, careful movement
```

#### `robot.move_backward(distance, speed)`
Move the robot backward by a specified distance.

**Parameters:**
- `distance` (float): Distance to move in meters
- `speed` (float): Movement speed in m/s

**Examples:**
```python
robot.move_backward(distance=2.0, speed=0.8)
robot.move_backward(distance=1.0, speed=0.5)
```

#### `robot.turn(direction, angle, speed)`
Rotate the robot in place.

**Parameters:**
- `direction` (str): Direction to turn - "left", "right", "clockwise", "counterclockwise"
- `angle` (int): Degrees to turn (30 - 360)
- `speed` (float): Rotation speed in rad/s

**Examples:**
```python
robot.turn(direction="left", angle=90, speed=0.5)
robot.turn(direction="clockwise", angle=180, speed=0.3)
robot.turn(direction="right", angle=45, speed=1.0)
```

#### `robot.move_to(x, y, z)`
Move to absolute coordinates in 3D space.

**Parameters:**
- `x` (float): X coordinate in meters
- `y` (float): Y coordinate in meters  
- `z` (float): Z coordinate in meters

**Examples:**
```python
robot.move_to(x=5.0, y=3.2, z=1.0)
robot.move_to(x=-2.5, y=4.7, z=0.5)
```

#### `robot.stop()`
Immediately stop all robot movement.

**Examples:**
```python
robot.stop()  # Emergency stop
```

---

### Gripper/Manipulation Commands

#### `robot.gripper.pick_up(object_type, color, size)`
Pick up an object using the gripper.

**Parameters:**
- `object_type` (str): Type of object - "cup", "ball", "box", "tool", etc.
- `color` (str): Object color - "red", "blue", "green", "yellow", "any", etc.
- `size` (str): Object size - "small", "medium", "large", "any"

**Examples:**
```python
robot.gripper.pick_up(object_type="cup", color="red", size="medium")
robot.gripper.pick_up(object_type="ball", color="blue", size="small")
robot.gripper.pick_up(object_type="box", color="any", size="large")
```

#### `robot.gripper.drop(location)`
Drop the currently held object at a location.

**Parameters:**
- `location` (str): Where to drop - "table", "floor", "shelf", "container", etc.

**Examples:**
```python
robot.gripper.drop(location="table")
robot.gripper.drop(location="red_container")
robot.gripper.drop(location="floor")
```

#### `robot.gripper.grab(object_name, force)`
Grab a specific object with controlled force.

**Parameters:**
- `object_name` (str): Specific object identifier
- `force` (float): Grip force in Newtons (0.1 - 5.0)

**Examples:**
```python
robot.gripper.grab(object_name="wrench", force=2.0)
robot.gripper.grab(object_name="fragile_vase", force=0.3)
robot.gripper.grab(object_name="heavy_tool", force=4.5)
```

#### `robot.gripper.release()`
Release/open the gripper to drop held objects.

**Examples:**
```python
robot.gripper.release()  # Open gripper
```

---

### Sensor Commands

#### `robot.sensors.scan(range, angle)`
Perform a sensor scan of the environment.

**Parameters:**
- `range` (float): Scan range in meters (1.0 - 20.0)
- `angle` (int): Scan angle in degrees (30 - 360)

**Examples:**
```python
robot.sensors.scan(range=10.0, angle=360)  # Full room scan
robot.sensors.scan(range=5.0, angle=180)   # Forward hemisphere
robot.sensors.scan(range=15.0, angle=90)   # Narrow cone scan
```

#### `robot.sensors.detect(object_type)`
Detect specific types of objects in the environment.

**Parameters:**
- `object_type` (str): Type to detect - "obstacles", "cups", "people", "objects", etc.

**Examples:**
```python
robot.sensors.detect(object_type="obstacles")
robot.sensors.detect(object_type="cups")
robot.sensors.detect(object_type="people")
robot.sensors.detect(object_type="red_objects")
```

#### `robot.sensors.measure_distance(direction)`
Measure distance to nearest object in a direction.

**Parameters:**
- `direction` (str): Direction to measure - "forward", "backward", "left", "right", "up", "down"

**Examples:**
```python
robot.sensors.measure_distance(direction="forward")
robot.sensors.measure_distance(direction="left")
robot.sensors.measure_distance(direction="up")
```

---

### Navigation Commands

#### `robot.navigate.go_to_room(room)`
Navigate to a specific room or area.

**Parameters:**
- `room` (str): Room name - "kitchen", "living_room", "bedroom", "office", etc.

**Examples:**
```python
robot.navigate.go_to_room(room="kitchen")
robot.navigate.go_to_room(room="living_room")
robot.navigate.go_to_room(room="bedroom")
robot.navigate.go_to_room(room="garage")
```

#### `robot.navigate.follow_path(path_name, speed)`
Follow a predefined path.

**Parameters:**
- `path_name` (str): Name of the path to follow
- `speed` (float): Speed to follow path in m/s

**Examples:**
```python
robot.navigate.follow_path(path_name="patrol_route", speed=1.0)
robot.navigate.follow_path(path_name="delivery_path", speed=0.8)
robot.navigate.follow_path(path_name="cleaning_route", speed=0.5)
```

#### `robot.navigate.return_home()`
Return to the robot's home/charging position.

**Examples:**
```python
robot.navigate.return_home()  # Go back to start position
```

---

## 🔧 Parameter Guidelines

### Distance Parameters
- **Short distances**: 0.1 - 1.0 meters
- **Medium distances**: 1.0 - 5.0 meters  
- **Long distances**: 5.0 - 10.0 meters

### Speed Parameters
- **Slow/careful**: 0.1 - 0.5 m/s
- **Normal**: 0.5 - 1.5 m/s
- **Fast**: 1.5 - 2.0 m/s

### Angle Parameters
- **Small turns**: 30 - 90 degrees
- **Medium turns**: 90 - 180 degrees
- **Large turns**: 180 - 360 degrees

### Force Parameters
- **Gentle**: 0.1 - 1.0 N (fragile objects)
- **Normal**: 1.0 - 3.0 N (regular objects)
- **Strong**: 3.0 - 5.0 N (heavy objects)

---

## 🎯 Common Usage Patterns

### Sequential Operations
For multi-step tasks, each command should be on a separate line:

```python
robot.navigate.go_to_room(room="kitchen")
robot.sensors.detect(object_type="cups")
robot.gripper.pick_up(object_type="cup", color="blue", size="medium")
robot.navigate.go_to_room(room="living_room")
robot.gripper.drop(location="coffee_table")
```

### Safety-First Operations
Always scan before moving in uncertain environments:

```python
robot.sensors.scan(range=5.0, angle=180)
robot.sensors.measure_distance(direction="forward")
robot.move_forward(distance=2.0, speed=0.5)
```

### Precision Manipulation
Use controlled force for delicate operations:

```python
robot.sensors.detect(object_type="fragile_items")
robot.gripper.grab(object_name="crystal_vase", force=0.2)
robot.move_to(x=3.0, y=2.0, z=1.2)
robot.gripper.drop(location="soft_surface")
```

---

## ❌ Common Mistakes to Avoid

1. **Invalid directions**: Use "left"/"right" or "clockwise"/"counterclockwise", not both
2. **Unrealistic parameters**: Keep distances under 10m, speeds under 2.0 m/s
3. **Missing parameters**: All functions require their specified parameters
4. **Invalid object types**: Use realistic object names like "cup", "book", "tool"
5. **Conflicting commands**: Don't move and turn simultaneously

---

## 🔄 API Extensions

To extend the API for specific robot platforms:

1. **Add new modules**: `robot.arm.*`, `robot.wheels.*`, etc.
2. **Add new parameters**: Additional safety or precision controls
3. **Add new object types**: Domain-specific objects
4. **Add new locations**: Environment-specific rooms/areas

Example extensions:
```python
# Robotic arm specific
robot.arm.extend(length=0.5)
robot.arm.retract(length=0.3)

# Lighting control
robot.lights.turn_on(brightness=0.8)
robot.lights.set_color(color="blue")

# Advanced sensors
robot.sensors.thermal_scan(range=5.0)
robot.sensors.chemical_detect(substance="gas")
```

This API provides a comprehensive foundation for natural language robot programming while remaining extensible for specific use cases.
