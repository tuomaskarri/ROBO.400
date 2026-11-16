import math
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def limit(value, maximum):
    return max(-maximum, min(maximum, value))


class GoalController(Node):

    def __init__(self):
        super().__init__('goal_controller')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.timer = self.create_timer(0.1, self.control)

        self.position = None
        self.goal = None
        self.waypoints = []
        self.circle = None
        self.running = True

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Convert quaternion to yaw angle
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.position = (p.x, p.y, theta)

    def stop(self):
        self.goal = None
        self.waypoints = []
        self.circle = None
        self.publisher.publish(Twist())
        self.get_logger().info("Robot stopped")

    def set_goal(self, x, y, theta=None):
        self.goal = (x, y, theta)
        self.waypoints = []
        self.circle = None

        if theta is None:
            self.get_logger().info(
                f"New point goal: x={x:.2f}, y={y:.2f}"
            )
        else:
            self.get_logger().info(
                f"New pose goal: x={x:.2f}, y={y:.2f}, "
                f"theta={math.degrees(theta):.1f} deg"
            )

    def set_square(self, side=1.0):
        if self.position is None:
            print("Waiting for odometry")
            return

        x, y, theta = self.position
        c = math.cos(theta)
        s = math.sin(theta)

        # Square corners relative to current position
        self.waypoints = [
            (x + side*c, y + side*s, None),
            (x + side*c - side*s, y + side*s + side*c, None),
            (x - side*s, y + side*c, None),
            (x, y, None)
        ]

        self.goal = self.waypoints.pop(0)
        self.circle = None

        self.get_logger().info(
            f"Starting square path, side={side:.2f} m"
        )

    def set_circle(self, radius=1.0):
        if self.position is None:
            print("Waiting for odometry")
            return

        x, y, theta = self.position

        # Circle center to the left of the robot
        cx = x - radius * math.sin(theta)
        cy = y + radius * math.cos(theta)

        self.stop()
        self.circle = (cx, cy, radius)

        self.get_logger().info(
            f"Starting circle path, radius={radius:.2f} m"
        )

    def control(self):
        if self.position is None:
            return

        x, y, theta = self.position
        cmd = Twist()

        # Continuously follow a circle
        if self.circle is not None:
            cx, cy, radius = self.circle

            phi = math.atan2(y - cy, x - cx)
            lookahead = 0.4

            gx = cx + radius * math.cos(phi + lookahead)
            gy = cy + radius * math.sin(phi + lookahead)

            distance = math.hypot(gx - x, gy - y)
            desired = math.atan2(gy - y, gx - x)
            error = wrap_angle(desired - theta)

            cmd.linear.x = min(0.15, 0.8 * distance)
            cmd.linear.x *= max(0.0, math.cos(error))
            cmd.angular.z = limit(2.0 * error, 1.0)

            self.publisher.publish(cmd)
            return

        if self.goal is None:
            self.publisher.publish(cmd)
            return

        gx, gy, gtheta = self.goal

        dx = gx - x
        dy = gy - y
        distance = math.hypot(dx, dy)

        # Move to the target position
        if distance > 0.07:
            desired = math.atan2(dy, dx)
            error = wrap_angle(desired - theta)

            # Rotate first if the heading error is large
            if abs(error) > 0.35:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = min(0.18, 0.6 * distance)

            cmd.angular.z = limit(1.5 * error, 1.0)

        # Rotate to target orientation
        elif gtheta is not None:
            error = wrap_angle(gtheta - theta)

            if abs(error) > 0.04:
                cmd.angular.z = limit(1.5 * error, 1.0)
            else:
                self.goal_reached()

        else:
            self.goal_reached()

        self.publisher.publish(cmd)

    def goal_reached(self):
        if self.waypoints:
            self.get_logger().info("Waypoint reached")
            self.goal = self.waypoints.pop(0)
            self.get_logger().info(
                f"Next waypoint: {self.goal}"
            )
        else:
            self.get_logger().info(
                "Goal reached successfully"
            )
            self.goal = None


def keyboard(node):
    print("\nCommands:")
    print("point x y")
    print("pose x y theta_degrees")
    print("square [side_length]")
    print("circle [radius]")
    print("stop")
    print("quit\n")

    while node.running:
        try:
            parts = input("Goal > ").split()

            if not parts:
                continue

            command = parts[0].lower()

            if command == 'point' and len(parts) == 3:
                node.set_goal(float(parts[1]), float(parts[2]))

            elif command == 'pose' and len(parts) == 4:
                angle = math.radians(float(parts[3]))
                node.set_goal(float(parts[1]), float(parts[2]), angle)

            elif command == 'square':
                side = float(parts[1]) if len(parts) > 1 else 1.0
                node.set_square(side)

            elif command == 'circle':
                radius = float(parts[1]) if len(parts) > 1 else 1.0
                node.set_circle(radius)

            elif command == 'stop':
                node.stop()

            elif command == 'quit':
                node.running = False

            else:
                print("Invalid command")

        except ValueError:
            print("Invalid number")
        except EOFError:
            node.running = False
        except KeyboardInterrupt:
            node.running = False


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = GoalController()

    thread = threading.Thread(target=keyboard, args=(node,), daemon=True)
    thread.start()

    try:
        while node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
