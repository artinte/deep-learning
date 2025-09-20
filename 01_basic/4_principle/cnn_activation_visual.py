import numpy
import os
from matplotlib import pyplot

# Set the Keras backend to "torch"
os.environ["KERAS_BACKEND"] = "torch"
import keras


def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    return numpy.expand_dims(array, axis=0)


# Build a ResNet50V2 model loaded with pre-trained ImageNet weights
model = keras.applications.ResNet50V2(weights="imagenet")
model.summary()

img_path = keras.utils.get_file(
    fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg"
)
img_tensor = get_img_array(img_path, target_size=(224, 224))

pyplot.axis("off")
pyplot.imshow(img_tensor[0].astype("uint8"))
pyplot.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)
pyplot.show()

# predict
predictions = model.predict(
    keras.applications.resnet_v2.preprocess_input(img_tensor.copy())
)
decoded_predictions = keras.applications.resnet_v2.decode_predictions(
    predictions, top=3
)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} ({score:.2f})")


# visual hidden layer
layer_outputs = []
layer_names = []
for index, layer in enumerate(model.layers):
    if isinstance(layer, keras.layers.Conv2D) and index % 2 == 0:
        layer_outputs.append(layer.output)
        layer_names.append(f"Layer {index+1} - {layer.name}")

activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

os.makedirs("temp", exist_ok=True)
fig, axs = pyplot.subplots(4, 6)
axs = axs.flatten()
for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
    if i >= 24:
        break
    print(activation.shape)
    im = axs[i].matshow(activation[0, :, :, 0])
    # axs[i].set_title(layer_name)
    axs[i].axis("off")
pyplot.tight_layout()
pyplot.savefig(os.path.join("temp", layer_name + ".png"))
pyplot.show()
pyplot.close()
