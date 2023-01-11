# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for exporting inference-only Keras models/layers."""

import tensorflow.compat.v2 as tf

from keras.engine import base_layer
from keras.engine import functional
from keras.engine import sequential


class ExportArchive(tf.__internal__.tracking.AutoTrackable):
    """ExportArchive is used to create inference-only SavedModel artifacts.

    If you have a Keras model or layer that you want to export as SavedModel for
    serving (e.g. via TensorFlow-Serving), you can use ExportArchive
    to configure the different serving endpoints you need to make available,
    as well as their signatures. Simply instantiate an ExportArchive,
    then use the `add_endpoint()` method to register a new serving endpoint.
    When done, use the `write_out()` method to save the artifact.

    The resulting artifact is a SavedModel and can be reloaded via
    `tf.saved_model.load`.

    Examples:

    Here's how to export a model for inference.

    ```python
    export_archive = ExportArchive(model)
    export_archive.add_endpoint(
        name="serve",
        fn=model.call,
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
    )
    export_archive.write_out("path/to/location")
    ```

    Here's how to export a model with one endpoint for inference and one
    endpoint for a training-mode forward pass (e.g. with dropout on).

    ```python
    export_archive = ExportArchive(model)
    export_archive.add_endpoint(
        name="call_inference",
        fn=lambda x: model.call(x, training=False),
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
    )
    export_archive.add_endpoint(
        name="call_training",
        fn=lambda x: model.call(x, training=True),
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
    )
    export_archive.write_out("path/to/location")
    ```
    """

    def __init__(self, layer):
        if not isinstance(layer, base_layer.Layer):
            raise ValueError(
                "Invalid layer type. Expected an instance of "
                "`keras.layers.Layer` or `keras.Model`. "
                f"Received instead an object of type '{type(layer)}'. "
                f"Object received: {layer}"
            )

        if not layer.built:
            raise ValueError(
                "The layer provided has not yet been built. "
                "It must be built before export."
            )

        self._trackables = list(layer._trackable_children().values())
        self.variables = list(layer.variables)
        self.trainable_variables = list(layer.trainable_variables)
        self.non_trainable_variables = list(layer.non_trainable_variables)
        self._endpoint_names = []

    def add_endpoint(self, name, fn, input_signature=None):
        """Register a new serving endpoint.

        Arguments:
            name: Str, name of the endpoint.
            fn: A function. It should only leverage resources (e.g. variables)
                that are available on the model/layer that was used to
                instantiate the ExportArchive.
                The shape and dtype of the inputs to the function must be
                known. For that purpose, you can either 1) make sure that
                `fn` is a `tf.function` that has been called at least once,
                2) provide an `input_signature` argument that specifies the
                shape and dtype of the inputs (see below).
            input_signature: Used to specify the shape and dtype of the
                inputs to `fn`. List of `tf.TensorSpec` objects.

        Example:

        Adding an endpoint using the `input_signature` argument:

        ```python
        export_archive = ExportArchive(model)
        export_archive.add_endpoint(
            name="my_endpoint",
            fn=model.call,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        ```

        Adding an endpoint that is a `tf.function`:

        ```python
        @tf.function()
        def my_endpoint(x):
            return model(x)

        # The function must be traced, i.e. it must be called at least once.
        my_endpoint(tf.random.normal(shape=(2, 3)))

        export_archive = ExportArchive(model)
        export_archive.add_endpoint(name="my_endpoint", fn=my_endpoint)
        ```
        """
        if name in self._endpoint_names:
            raise ValueError(f"Endpoint name '{name}' is already taken.")

        if input_signature:
            decorated_fn = tf.function(fn, input_signature=input_signature)
        else:
            if isinstance(fn, tf.types.experimental.GenericFunction):
                if not fn._list_all_concrete_functions():
                    raise ValueError(
                        f"The provided tf.function '{fn}' "
                        "has never been called. "
                        "To specify the expected shape and dtype "
                        "of the function's arguments, "
                        "you must either provide a function that "
                        "has been called at least once, or alternatively pass "
                        "an `input_signature` argument in `add_endpoint()`."
                    )
                decorated_fn = fn
            else:
                raise ValueError(
                    "If the `fn` argument provided is not a tf.function, "
                    "you must provide an `input_signature` argument to "
                    "specify the shape and dtype of the function arguments. "
                    "Example:\n\n"
                    "export_archive.add_endpoint(\n"
                    "    name='call',\n"
                    "    fn=model.call,\n"
                    "    input_signature=[\n"
                    "        tf.TensorSpec(\n"
                    "            shape=(None, 224, 224, 3),\n"
                    "            dtype=tf.float32,\n"
                    "        )\n"
                    "    ],\n"
                    ")"
                )
        setattr(self, name, decorated_fn)
        self._endpoint_names.append(name)

    def add_variable_collection(self, name, variables):
        """Register a set of variables to be retrieved after reloading.

        Arguments:
            name: The string name for the collection.
            variables: A tuple/list/set of `tf.Variable` instances.

        Example:

        ```python
        export_archive = ExportArchive(model)
        # Register an endpoint
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        # Save a variable collection
        export_archive.add_variable_collection(
            name="optimizer_variables", variables=model.optimizer.variables)
        export_archive.write_out("path/to/location")

        # Reload the object
        revived_object = tf.saved_model.load("path/to/location")
        # Retrieve the variables
        optimizer_variables = revived_object.optimizer_variables
        ```
        """
        if not isinstance(variables, (list, tuple, set)):
            raise ValueError(
                "Expected `variables` to be a list/tuple/set. "
                f"Received instead object of type '{type(variables)}'."
            )
        if not all(isinstance(v, tf.Variable) for v in variables):
            raise ValueError(
                "Expected all elements in `variables` to be "
                "`tf.Variable` instances. Found instead the following types: "
                f"{list(set(type(v) for v in variables))}"
            )
        setattr(self, name, list(variables))

    def write_out(self, filepath, options=None):
        """Write the corresponding SavedModel to disk.

        Arguments:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the artifact.
            options: `tf.saved_model.SaveOptions` object that specifies
                SavedModel saving options.
        """
        if not self._endpoint_names:
            raise ValueError(
                "No endpoints have been set yet. Call add_endpoint()."
            )
        tf.saved_model.save(self, filepath, options=options)
        endpoints = "\n\n".join(
            _print_signature(getattr(self, name), name)
            for name in self._endpoint_names
        )
        print(
            f"Saved artifact at '{filepath}'. "
            "The following endpoints are available:\n\n"
            f"{endpoints}"
        )


def export_model(model, filepath):
    export_archive = ExportArchive(model)
    if isinstance(model, (functional.Functional, sequential.Sequential)):
        input_signature = tf.nest.map_structure(_make_tensor_spec, model.inputs)
        export_archive.add_endpoint("serve", model.__call__, input_signature)
    else:
        save_spec = model._get_save_spec()
        if not save_spec:
            raise ValueError(
                "The model provided has never called. "
                "It must be called at least once before export."
            )
        input_signature = [save_spec]
        export_archive.add_endpoint("serve", model.__call__, input_signature)
    export_archive.write_out(filepath)


def _make_tensor_spec(x):
    return tf.TensorSpec(x.shape, dtype=x.dtype)


def _print_signature(fn, name):
    concrete_fn = fn._list_all_concrete_functions()[0]
    pprinted_signature = concrete_fn.pretty_printed_signature(verbose=True)
    lines = pprinted_signature.split("\n")
    lines = [f"* Endpoint '{name}'"] + lines[1:]
    endpoint = "\n".join(lines)
    return endpoint
