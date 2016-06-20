package org.ai.layer.impl;

import org.ai.function.TransferFunctionPair;
import org.ai.function.activations.Activation;
import org.ai.function.derivations.Derivation;
import org.ai.layer.Layer;

import java.util.Arrays;
import java.util.Optional;

public final class FullyConnectedLayer extends Layer {

    public FullyConnectedLayer(int size, int connections, long seed, TransferFunctionPair pair) {
        super(size, connections, seed, pair);
    }

    public FullyConnectedLayer(float[][] weights, TransferFunctionPair pair) {
        super(weights, pair);
    }

    @Override
    public final float[] forward(float[] input) {
        float[] results = new float[weights.length];
        for (int j = 0; j < weights.length; j++) {
            for (int k = 0; k < weights[j].length; k++) {
                results[j] += input[k] * weights[j][k];
            }
        }
        return activate(results);
    }

    public static <A extends Activation, D extends Derivation> Optional<Layer> of(Class<A> activationClass, Class<D> deactivationClass, int size, int connections, long seed) {
        try {
            A activation = activationClass.newInstance();
            D deactivation = deactivationClass.newInstance();
            return Optional.of(new FullyConnectedLayer(size, connections, seed, new TransferFunctionPair<>(activation, deactivation)));
        } catch (InstantiationException | IllegalAccessException e) {
            e.printStackTrace();
        }
        return Optional.empty();
    }

    @Override
    public final String toString() {
        return "(FullyConnectedLayer weights: " + Arrays.deepToString(weights) + ")";
    }
}
