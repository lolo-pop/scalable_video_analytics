import os
import logging
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

# from tensorflow.compat.v1 import ConfigProto
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
par = {'1': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/pool1/MaxPool',
       '2': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block2/unit_1/bottleneck_v1/Relu',
       '3': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_1/bottleneck_v1/Relu',
       '4': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_23/bottleneck_v1/Relu',
       '5': 'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/Relu'
}
from numba import cuda



class Detector:
    classes = {
        "vehicle": [3, 6, 7, 8],
        "persons": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]
    }
    rpn_threshold = 0.5

    def __init__(self, model_path='frozen_inference_graph.pb'):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        tf.disable_v2_behavior()
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.05
        self.model_path = model_path
        self.d_graph = tf.Graph()
        with self.d_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.Session(config=config)
        self.logger.info("Object detector initialized")

    def run_inference_for_single_image(self, image, graph, point):
        with self.d_graph.as_default():
            # Get handles to input and output tensors
            # ops = tf.get_default_graph().get_operations()
            # all_tensor_names = {output.name for op in ops for output in op.outputs}

            image_tensor = (tf.get_default_graph()
                            .get_tensor_by_name('image_tensor:0'))
            feed_dict = {image_tensor: np.expand_dims(image, 0)}
            # FOR RCNN final layer results:
            tensor_dict = []

            boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
            scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
            classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
            num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')
            '''
            for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    # 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.get_default_graph().get_tensor_by_name(tensor_name))

            # FOR RPN intermedia results
            key_tensor_map = {
                "RPN_box_no_normalized": ("BatchMultiClassNonMaxSuppression"
                                          "/map/while/"
                                          "MultiClassNonMaxSuppression/"
                                          "Gather/Gather:0"),
                "RPN_score": ("BatchMultiClassNonMaxSuppression/"
                              "map/while/"
                              "MultiClassNonMaxSuppression"
                              "/Gather/Gather_2:0"),
                "Resized_shape": ("Preprocessor/map/while"
                                  "/ResizeToRange/stack_1:0"),
            }

            for key, tensor_name in key_tensor_map.items():
                if tensor_name in all_tensor_names:
                    tensor_dict[tensor_name] = (
                        tf.get_default_graph()
                        .get_tensor_by_name(tensor_name))
            '''
            output_dict = self.session.run([boxes, scores, classes, num_detections],
                                           feed_dict=feed_dict)

            # Run inference


            '''
            # FOR RPN intermedia results
            w = output_dict[key_tensor_map['Resized_shape']][1]
            h = output_dict[key_tensor_map['Resized_shape']][0]
            input_shape_array = np.array([h, w, h, w])
            output_dict['RPN_box_normalized'] = output_dict[key_tensor_map[
                'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]
            output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']]

            # FOR RCNN final layer results
            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = (
                output_dict['detection_boxes'][0])
            output_dict['detection_scores'] = (
                output_dict['detection_scores'][0])
            '''
        return output_dict
    def close(self):
        self.logger.info('clear the gpu memory')
        self.session.close()
        device = cuda.get_current_device()
        cuda.close()

    def infer(self, image_np, point):
        imgae_crops = image_np

        # this output_dict contains both final layer results and RPN results
        output_dict = self.run_inference_for_single_image(
            imgae_crops, self.d_graph, point)
        # print(output_dict)
        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        results = []
        rpn_results = []
        # for i range(len(output_dict['detection_boxes'])):
        #     object_class = output_dict['detection_classes'][i]
        '''
        for i in range(len(output_dict['detection_boxes'])):
            object_class = output_dict['detection_classes'][i]
            # print(type(object_class))
            relevant_class = False
            for k in Detector.classes.keys():
                if object_class in Detector.classes[k]:
                    object_class = k
                    relevant_class = True
                    break
            if not relevant_class:
                continue
            if  output_dict['detection_scores'][i] > 0:
                ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
                confidence = output_dict['detection_scores'][i]
                box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
                fpn = output_dict['RPN_box_normalized'][i]
                fpn_score = output_dict['RPN_score'][i]
                results.append((object_class, confidence, box_tuple))
                rpn_results.append((object_class, box_tuple, fpn, fpn_score))

        # Get RPN regions along with classification results
        # rpn results array will have (class, (xmin, xmax, ymin, ymax)) typles
    
        results_rpn = []
        for idx_region, region in enumerate(output_dict['RPN_box_normalized']):
            x = region[1]
            y = region[0]
            w = region[3] - region[1]
            h = region[2] - region[0]
            conf = output_dict['RPN_score'][idx_region]
            if conf < Detector.rpn_threshold or w * h == 0.0 or w * h > 0.04:
                continue
            results_rpn.append(("object", conf, (x, y, w, h)))
        '''
        # return results, rpn_results

        return output_dict