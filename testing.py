from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, \
    Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid, json, math
from dataclasses import dataclass


@dataclass
class Batch():
    images: list
    batch_count: int


image_count_per_batch = 51
batch_count = 2

# Replace with valid values
with open("credentials.json") as f:
    data = json.load(f)
    endpoint_training = data["endpoint_training"]
    endpoint_prediction = data["endpoint_prediction"]
    training_key = data["training_key"]
    prediction_key = data["prediction_key"]
    prediction_resource_id = data["prediction_resource_id"]

print("Gotten credentials")

# Instantiate training and prediction client
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(endpoint_training, credentials)

print("Instantiated training")

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint_prediction, prediction_credentials)

print("Instantiated prediction")

publish_iteration_name = "custom-vision-ORBIT-testing"

print("Creating project")
project_name = "Testing-" + str(uuid.uuid4())

project_list = trainer.get_projects()

for project in project_list:
    trainer.delete_project(project.id)

project = trainer.create_project(project_name)

print("Crawling dataset")
base_path = os.getcwd() + "/dataset/validation"

os.chdir(base_path)

cluttered_list = []

users = os.listdir()
batch_num = 0

for user in users:
    print("Crawling user: " + user)
    datasets = os.listdir(base_path + "/" + user)

    for dataset in datasets:
        batch_num += 1
        if batch_num > batch_count:
            break

        print("Crawling dataset: " + dataset)
        print("Making tag")

        dataset_tag = trainer.create_tag(project.id, dataset)

        os.chdir(base_path + "/" + user + "/" + dataset)

        clean = base_path + "/" + user + "/" + dataset + "/clean"
        cluttered = base_path + "/" + user + "/" + dataset + "/clutter"
        if not os.path.exists(clean):
            print("No clean directory, skipping")
            continue

        if not os.path.exists(cluttered):
            print("No cluttered directory, skipping")
            continue

        image_count = 0

        for subdataset in os.listdir(clean):
            image_count += len(os.listdir(clean + "/" + subdataset))

        step = math.floor(image_count / image_count_per_batch)
        i = 0

        batches = []
        batches.append(Batch(images=[], batch_count=0))

        for subdataset in os.listdir(clean):
            for image in os.listdir(clean + "/" + subdataset):
                i += 1
                if i == step:
                    i = 0

                    with open(clean + "/" + subdataset + "/" + image, mode="rb") as f:
                        if batches[-1].batch_count >= 64:
                            batches.append(Batch(images=[], batch_count=0))

                        batches[-1].images.append(
                            ImageFileCreateEntry(name=image, contents=f.read(), tag_ids=[dataset_tag.id]))
                        batches[-1].batch_count += 1
                        print("Added image: " + image)

        print("Batch count: " + str(len(batches)))
        for batch in batches:
            print("Creating batch")
            upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=batch.images))
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status)
                exit(-1)

            print("Batch created")

        print("Batch upload complete")

        # make a list of things in the cluttered subdatasets so we can go over them later
        cluttered_subdatasets = os.listdir(cluttered)
        for cluttered_subdataset in cluttered_subdatasets:
            for image in os.listdir(cluttered + "/" + cluttered_subdataset):
                cluttered_list.append(cluttered + "/" + cluttered_subdataset + "/" + image)

print("Training project")
iteration = trainer.train_project(project.id)

startTime = time.time()

while True:
    iteration = trainer.get_iteration(project.id, iteration.id)
    print("Training status: " + iteration.status)
    if iteration.status == "Completed":
        break

    print("Waiting 15 seconds... it has been " + str(math.round(time.time() - startTime)) + " seconds")
    time.sleep(15)

print("Training complete. Iteration ID: " + str(iteration.id))
print("Iteration publish name: " + str(publish_iteration_name))
print("Iteration public name [2]:" + str(iteration.publish_name))
print("Iteration resource id: " + str(prediction_resource_id))

print("Publishing!")
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print("Done!")

for path in cluttered_list:
    with open(path, mode="rb") as f:
        results = predictor.classify_image(project.id, publish_iteration_name, f.read())

        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))
