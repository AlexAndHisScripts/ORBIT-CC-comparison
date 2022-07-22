# Azure stuff
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials

# Image conversion
from io import BytesIO
from PIL import Image

# Misc
import os, time, json, math

# Data structures
from dataclasses import dataclass

# CLI library
import click


@dataclass
class Credentials:
    endpoint_training: str
    endpoint_prediction: str
    training_key: str
    prediction_key: str
    prediction_resource_id: str


@dataclass
class Batch:
    images: list
    count: int


def get_credentials() -> Credentials:
    if not os.path.exists("credentials.json"):
        with open("credentials.json", "w") as f:
            json.dump({"endpoint_training": "", "endpoint_prediction": "", "training_key": "", "prediction_key": "", "prediction_resource_id": ""}, f, indent=4)

        print("Created credentials.json file. Please fill it in.")
        return

    with open("credentials.json") as f:
        data = json.load(f)
        endpoint_training = data["endpoint_training"]
        endpoint_prediction = data["endpoint_prediction"]
        training_key = data["training_key"]
        prediction_key = data["prediction_key"]
        prediction_resource_id = data["prediction_resource_id"]

    return Credentials(endpoint_training, endpoint_prediction, training_key, prediction_key, prediction_resource_id)


def get_trainer(credentials: Credentials) -> CustomVisionTrainingClient:
    apikey = ApiKeyCredentials(in_headers={"Training-key": credentials.training_key})
    return CustomVisionTrainingClient(credentials.endpoint_training, apikey)


def get_predictor(credentials: Credentials) -> CustomVisionPredictionClient:
    apikey = ApiKeyCredentials(in_headers={"Prediction-key": credentials.prediction_key})
    return CustomVisionPredictionClient(credentials.endpoint_prediction, apikey)


def get_clients(credentials: Credentials) -> (CustomVisionTrainingClient, CustomVisionPredictionClient):
    return get_trainer(credentials), get_predictor(credentials)


def scale_image(img_data, size):
    def image_to_byte_array(image: Image) -> bytes:
        imgByteArr = BytesIO()
        image.save(imgByteArr, "JPEG")
        imgByteArr.seek(0)
        return imgByteArr.read()

    try:
        int(size)
    except:
        return img_data

    size = int(size)

    image = Image.open(img_data)

    return image_to_byte_array(image.resize((size, size), Image.Resampling.NEAREST))


@click.group()
def cli():
    pass


@click.command()
def gen_creds():
    get_credentials()

@click.command()
def list_projects():
    trainer = get_trainer(get_credentials())
    projects = trainer.get_projects()

    if len(projects) == 0:
        print("No projects found")
        return

    print("Projects:")
    for project in trainer.get_projects():
        print("\t" + project.name)
        print("\t\t | ID:      " + project.id)

        count = trainer.get_image_count(project.id)
        print("\t\t | Images:  " + str(count))

        print("\t\t | Iterations: " + str(len(trainer.get_iterations(project.id))))


@click.command()
@click.option("--name", prompt="Project name", help="Name of the project")
def create_project(name: str):
    trainer = get_trainer(get_credentials())

    amount_of_projs = len(trainer.get_projects())
    if amount_of_projs == 2:
        print("You can only have 2 projects at once. Delete a project to continue.")

    project = trainer.create_project(name)
    print("Project created: " + project.name + " (" + project.id + ")")


@click.command()
@click.option("--name", prompt="Project name", help="Name of the project")
def delete_project(name: str):
    trainer = get_trainer(get_credentials())
    projects = trainer.get_projects()

    for project in projects:
        if project.name == name:
            trainer.delete_project(project.id)

            print("Project deleted: " + project.name + " (" + project.id + ")")
            return

    print("Project not found")


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
@click.option("--dataset-count", prompt="Dataset count", help="Number of datasets to process.")
@click.option("--images-per-dataset", prompt="Image count", help="Number of images to process per dataset.")
@click.option("--cluttered-filename", prompt="Cluttered filename", help="Path to write the cluttered list to")
@click.option("--img-size", prompt="Image size", help="Dimensions that images will be scaled to (square)")
def upload_images(project_id, dataset_count: int, images_per_dataset: int, cluttered_filename, img_size):
    dataset_count = int(dataset_count)
    images_per_dataset = int(images_per_dataset)

    print("Getting trainer")
    trainer = get_trainer(get_credentials())

    print("Removing images")
    trainer.delete_images(project_id, all_images=True, all_iterations=True)

    cluttered_images = []
    dataset_num = 0

    taglist = trainer.get_tags(project_id)

    for tag in taglist:
        trainer.delete_tag(project_id, tag.id)

    users = os.listdir("dataset/validation")
    for user in users:
        if dataset_num > dataset_count:
            break

        print("Processing " + user)
        for dataset in os.listdir("dataset/validation/" + user):
            dataset_num += 1
            if dataset_num > dataset_count:
                break

            print("Creating tag for dataset: " + dataset)

            try:
                tag = trainer.create_tag(project_id, dataset)
            except:
                print("Tag already exists")
                found = False
                for othertag in trainer.get_tags(project_id):
                    if othertag.name == dataset:
                        tag = othertag
                        found = True
                        break

                if not found:
                    # Just ignore this dataset
                    break

            clean = "dataset/validation/" + user + "/" + dataset + "/clean"
            cluttered = "dataset/validation/" + user + "/" + dataset + "/clutter"

            image_count = 0

            for subdataset in os.listdir(clean):
                image_count += len(os.listdir(clean + "/" + subdataset))

            step = math.floor(image_count / images_per_dataset)
            i = 0

            batches = [Batch([], 0)]
            for subdataset in os.listdir(clean):
                for image in os.listdir(clean + "/" + subdataset):
                    i += 1
                    if i == step:
                        i = 0

                        with open(clean + "/" + subdataset + "/" + image, mode="rb") as f:
                            if batches[-1].count >= 64:
                                batches.append(Batch(images=[], batch_count=0))

                            batches[-1].images.append(
                                ImageFileCreateEntry(name=image, contents=scale_image(f, img_size), tag_ids=[tag.id]))

                            batches[-1].count += 1
                            print("Added image: " + image)

            print("Batch count: " + str(len(batches)))
            for batch in batches:
                print("Creating batch")
                upload_result = trainer.create_images_from_files(project_id, ImageFileCreateBatch(images=batch.images))
                if not upload_result.is_batch_successful:
                    print("Image batch upload failed.")
                    can_recover = True
                    for image in upload_result.images:
                        if image.status != "OK" and image.status != "OKDuplicate":
                            print("Image status: ", image.status)
                            can_recover = False

                    if not can_recover:
                        exit(-1)

                print("Batch created")

            print("Batch upload complete")

            for subdataset in os.listdir(cluttered):
                for image in os.listdir(cluttered + "/" + subdataset):
                    cluttered_images.append(cluttered + "/" + subdataset + "/" + image)

    print("Writing cluttered images")
    with open(cluttered_filename + ".clutteredimgs", "w") as f:
        first = True
        for image in cluttered_images:
            if first:
                f.write(image)
                first = False

            f.write("\n" + image)


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
def train(project_id):
    print("Training")
    trainer = get_trainer(get_credentials())
    iteration = trainer.train_project(project_id)
    print("Started training iteration: " + str(iteration.id))
    print("Recommend usage of the show-iteration-status and list-iterations commands.")
    with open(iteration.id + ".iterationinfo", "w") as f:
        f.write(json.dumps({"starttime": time.time()}, indent=4, sort_keys=True))


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
@click.option("--iteration-id", prompt="Iteration ID", help="ID of the iteration")
def show_iteration_status(project_id, iteration_id):
    trainer = get_trainer(get_credentials())
    with open(iteration_id + ".iterationinfo", "r") as f:
        data = json.loads(f.read())
        starttime = float(data["starttime"])

    while True:
        iteration = trainer.get_iteration(project_id, iteration_id)
        timeTaken = time.time() - starttime

        print(iteration.status + " (Started " + str(round(timeTaken / 60, 1)) + "m ago)")

        if iteration.status == "Completed":
            return

        last = str(round(timeTaken / 60, 1))

        while str(round(timeTaken / 60, 1)) == last:
            time.sleep(1)
            timeTaken = time.time() - starttime


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
def list_iterations(project_id):
    trainer = get_trainer(get_credentials())

    project_name = None
    for project in trainer.get_projects():
        if project.id == project_id:
            project_name = project.name

    iterations = trainer.get_iterations(project_id)

    if len(iterations) == 0:
        print("No iterations found")
        return

    print(f"Iterations for project \"{project_name}\" ({project_id})")

    for iteration in iterations:
        print("\t" + iteration.name)
        print(f"\t\t | ID:         {iteration.id}")
        print(f"\t\t | Status:     {iteration.status}")
        print(f"\t\t | Published:  {iteration.publish_name}")


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
@click.option("--iteration-id", prompt="Iteration ID", help="ID of the iteration")
def delete_iteration(project_id, iteration_id):
    trainer = get_trainer(get_credentials())

    trainer.delete_iteration(project_id, iteration_id)
    print("Done!")


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
@click.option("--iteration-id", prompt="Iteration ID", help="ID of the iteration")
@click.option("--publish-iteration-name", prompt="Publish Iteration Name",
              help="Name of the iteration that you are publishing")
def publish_iteration(project_id, iteration_id, publish_iteration_name):
    print(f"Publishing iteration under name {publish_iteration_name}")

    creds = get_credentials()
    trainer = get_trainer(creds)

    trainer.publish_iteration(project_id, iteration_id, publish_iteration_name, creds.prediction_resource_id)

    print("Published!")


@click.command()
@click.option("--project-id", prompt="Project ID", help="ID of the project")
@click.option("--publish-iteration-name", prompt="Publish Iteration Name",
              help="Publish-iteration-name of the iteration you want to export for")
@click.option("--cluttered-file", prompt="Cluttered file name",
              help="Filename of the file that contains cluttered paths")
@click.option("--outfile", prompt="Output filename", help="Full output filename")
@click.option("--amount", prompt="Amount of images to test", help="Amount of images to test on")
@click.option("--img-size", prompt="Size of image (scale to)", help="The size the images are scaled to")
def process_and_export_stats(project_id, publish_iteration_name, cluttered_file, outfile, amount, img_size):
    def make_json_of_predictions(predictions):
        out = []
        for i in predictions:
            out.append({"prediction": i.tag_name, "probability": i.probability})

        return out

    predictor = get_predictor(get_credentials())
    output = []
    amount = int(amount)

    with open(cluttered_file, "r") as f:
        paths = f.read().split("\n")

        interval = math.floor((len(paths) / amount))

        print("Interval is " + str(interval))

        i = 0

        taglist = []

        for path in paths:
            i += 1
            if i % 10 == 0:
                print(str(i) + " : " + str(interval) + " : " + str(amount) + " : " + str(len(paths)))

            if i % interval == 0:
                print(path)

                with open(path, mode="rb") as image:
                    results = predictor.classify_image(project_id, publish_iteration_name, scale_image(image, img_size))

                    tag = path.split("/")[3]

                    prob = None

                    # log if tag is highest one
                    # average accuracy for users
                    highestProb = None
                    highestProbValue = 0

                    for prediction in results.predictions:
                        if prediction.probability > highestProbValue:
                            highestProbValue = prediction.probability
                            highestProb = prediction.tag_name

                        if not prediction.tag_name in taglist:
                            taglist.append(prediction.tag_name)

                        if prediction.tag_name == tag:
                            prob = prediction.probability
                            break

                    if not prob:
                        print("Invalid prob")
                        continue

                    output.append({"tag": tag, "returnedprob": prob, "highestprob": highestProb,
                                   "highestprobvalue": highestProbValue,
                                   "raw_predict": make_json_of_predictions(results.predictions)})

        print("Writing")
        with open(outfile, "w") as f:
            f.write(json.dumps({"data": output, "meta": {"taglist": taglist}}, indent=4))

        print("Done")


@click.command()
@click.option("--infile", prompt="Input filename", help="Full input filename")
@click.option("--outprefix", prompt="Output prefix", help="Prefix for outputted files")
def json_to_histogram(infile, outprefix):
    import matplotlib.pyplot as plt

    def calc_average(dataset):
        total = 0
        for i in dataset:
            total += i

        return total / len(dataset)

    def calc_median(dataset):
        dataset.sort()
        if len(dataset) % 2 == 0:
            return (dataset[len(dataset) // 2] + dataset[len(dataset) // 2 - 1]) / 2
        else:
            return dataset[len(dataset) // 2]

    with open(infile, "r") as f:
        data = json.loads(f.read())
        taglist = data["meta"]["taglist"]

        histograms = {}
        amountHighestProb = {}

        successRates = []

        for tag in taglist:
            for entry in data["data"]:
                if entry["tag"] == tag:
                    if tag not in histograms:
                        histograms[tag] = []

                    histograms[tag].append(entry["returnedprob"])

                    if entry["highestprob"] == tag:
                        if tag not in amountHighestProb:
                            amountHighestProb[tag] = {}
                            amountHighestProb[tag]["positives"] = 1
                            amountHighestProb[tag]["total"] = 1

                        amountHighestProb[tag]["positives"] += 1
                        amountHighestProb[tag]["total"] += 1

                    else:
                        if tag not in amountHighestProb:
                            amountHighestProb[tag] = {}
                            amountHighestProb[tag]["positives"] = 0
                            amountHighestProb[tag]["total"] = 1

                        amountHighestProb[tag]["total"] += 1

        for name, values in histograms.items():
            print(name)
            highestProbPercent = amountHighestProb[name]["positives"] / amountHighestProb[name]["total"]
            print(f"Highest prob: {highestProbPercent}")

            successRates.append(highestProbPercent)

            plt.close()
            plt.xlim([0, 1])
            plt.hist(values, bins=[x / 20 for x in range(0, 21)], label=name)
            plt.title(name + " (Success rate: " + str(round(highestProbPercent * 100, 2)) + "%)")
            # plt.show()
            plt.savefig(outprefix + "_" + name.replace(" ", "") + ".png")

        print("Mean success rate: " + str(calc_average(successRates) * 100) + "%")
        print("Median success rate: " + str(calc_median(successRates) * 100) + "%")


if __name__ == "__main__":
    cli.add_command(list_projects)
    cli.add_command(create_project)
    cli.add_command(delete_project)
    cli.add_command(upload_images)
    cli.add_command(train)
    cli.add_command(show_iteration_status)
    cli.add_command(list_iterations)
    cli.add_command(delete_iteration)
    cli.add_command(publish_iteration)
    cli.add_command(process_and_export_stats)
    cli.add_command(json_to_histogram)
    cli.add_command(gen_creds)

    cli()
