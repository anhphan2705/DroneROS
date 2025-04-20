#!/usr/bin/env python3
import yaml
import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from msgs.msg import TrackedBoundingBoxes, TrackedBoundingBox

class UniversalIDMapper(Node):
    def __init__(self):
        super().__init__('id_mapper_node')

        # parameter to point at your YAML
        self.declare_parameter('tracked_topic', '/yolo/detections_0/depth/tracked/classified/light')
        self.declare_parameter('map_topic', '/yolo/detections_0/depth/tracked/classified/light/mapped')
        self.declare_parameter('mapping_file', 'mapping.yaml')
        mapping_file = self.get_parameter('mapping_file').value
        self.tracked_topic = self.get_parameter('tracked_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        
        if os.path.isabs(mapping_file):
            self.mapping_path = mapping_file
        else:
            pkg_dir = get_package_share_directory('perception')
            self.mapping_path = os.path.join(pkg_dir, "models", mapping_file)
        if not os.path.exists(self.mapping_path):
            self.get_logger().error(f"Mapping file not found: {self.model_path}")
            rclpy.shutdown()
            return

        # load up all the (class_id, classification_id) → universal_id entries
        try:
            with open(self.mapping_path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().fatal(f'Couldn’t load mapping file: {e}')
            rclpy.shutdown()
            return

        # build lookup: (class_id, classification_id) → {id, class_name}
        self._map = {
            (m['class_id'], m['classification_id']): 
            {
                'id':         m['id'],
                'class_name': m['class_name']
            }
            for m in data.get('mappings', [])
        }
        self.get_logger().info(f'Loaded {len(self._map)} mappings')

        # subscribe to original tracked boxes
        self.sub = self.create_subscription(
            TrackedBoundingBoxes,
            self.tracked_topic,
            self.callback,
            10,
        )
        # publish remapped boxes
        self.pub = self.create_publisher(
            TrackedBoundingBoxes,
            self.map_topic,
            10,
        )

    def callback(self, msg: TrackedBoundingBoxes):
        out = TrackedBoundingBoxes()
        out.header = msg.header

        for box in msg.boxes:
            key = (box.class_id, box.classification_id)
            entry = self._map.get(key)
            if entry:
                box.id         = entry['id']
                box.class_name = entry['class_name']
            else:
                self.get_logger().warn(
                    f"No mapping for class_id={box.class_id}, "
                    f"classification_id={box.classification_id}"
                )
            out.boxes.append(box)

        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = UniversalIDMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()