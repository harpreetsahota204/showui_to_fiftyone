# Import required libraries
from datasets import load_dataset
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Load dataset from HuggingFace hub
dataset = fouh.load_from_hub(
    "showlab/ShowUI-desktop",
    name="ShowUI_desktop",
    format="ParquetFilesDataset",
    split="train",
    overwrite=True,
)

def convert_to_fiftyone_classifications(labels):
    """
    Convert strings to FiftyOne Classification objects.

    Args:
    labels (list): List of strings representing classification labels

    Returns:
    list: List of FiftyOne Classification objects
    """
    classifications = []

    for label in labels:
        # Create Classification with the label string
        classification = fo.Classification(
            label=label
        )
        classifications.append(classification)

    return classifications


def convert_to_fiftyone_detections(bbox_coords):
    """
    Convert normalized bounding box coordinates to FiftyOne Detection objects.

    Takes bounding boxes in [x1, y1, x2, y2] format (normalized coordinates where x1,y1 is top-left 
    and x2,y2 is bottom-right) and converts them to FiftyOne's [x, y, width, height] format 
    (where x,y is top-left and width,height are the box dimensions).

    Args:
        bbox_coords (list): List of lists, where each inner list contains 4 float values 
            representing normalized coordinates [x1, y1, x2, y2] of a bounding box

    Returns:
        list: List of FiftyOne Detection objects, each containing:
            - label: "action_button" 
            - bounding_box: [x, y, width, height] in normalized coordinates
    """
    detections = []

    for box in bbox_coords:
        x1, y1, x2, y2 = box
        
        # Calculate width and height from diagonal points
        # width = right_x - left_x
        # height = bottom_y - top_y
        width = x2 - x1  
        height = y2 - y1
        
        # Create Detection with normalized coordinates:
        # x,y: top-left corner position
        # width,height: box dimensions
        detection = fo.Detection(
            label="action",
            bounding_box=[x1, y1, width, height]
        )

        detections.append(detection)

    return detections


def convert_to_fiftyone_keypoints(points):
    """
    Convert normalized [x, y] coordinates to FiftyOne Keypoints objects.

    Args:
    points (list of lists): List of point coordinates in format [x, y]

    Returns:
    list: List of FiftyOne Keypoints objects
    """
    keypoints_list = []

    for point in points:
        x, y = point
        
        # Create FiftyOne Keypoints object
        # For each keypoint, points should be a list of [x,y] pairs
        keypoints = fo.Keypoints(
            keypoints=[fo.Keypoint(
                label="action",
                points=[[x, y]]  # Note the double brackets here - list of [x,y] pairs
            )]
        )
        
        keypoints_list.append(keypoints)

    return keypoints_list


def main():
    # Load the Parquet file into a pandas DataFrame
    df = pd.read_parquet('ShowUI-web/data/train-00000-of-00001.parquet')
    
    # Set base directory
    base_dir = "/home/harpreet/workspace/show-ui"
    
    # Create and process the dataset
    dataset = create_fiftyone_dataset(df, base_dir)
    
    # Convert and set values in the dataset
    bbox_coords = dataset.values('bbox')
    detections = convert_to_fiftyone_detections(bbox_coords)
    dataset.set_values("action_detections", detections)

    point_coords = dataset.values('point')
    keypoints = convert_to_fiftyone_keypoints(point_coords)
    dataset.set_values("action_keypoints", keypoints)

    query_types = dataset.values('type')
    queries = convert_to_fiftyone_classifications(query_types)
    dataset.set_values("query_type", queries)

    folder_path = [path.split('/')[1] for path in dataset.values("image_url")]
    interfaces = convert_to_fiftyone_classifications(folder_path)
    dataset.set_values("interfaces", interfaces)

    # Clean up and prepare dataset
    dataset.delete_sample_fields(['image_url', 'bbox', 'point', 'type', 'row_idx'])
    dataset.add_dynamic_sample_fields()
    dataset = dataset.shuffle()
    dataset.compute_metadata()
    dataset.save()

if __name__ == "__main__":
    main()