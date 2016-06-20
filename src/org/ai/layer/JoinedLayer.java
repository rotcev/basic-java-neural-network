package org.ai.layer;

import org.ai.layer.impl.FullyConnectedLayer;

import java.util.Arrays;
import java.util.Optional;

public final class JoinedLayer {
    private final Optional<Layer> previousLayer;
    private final Optional<Layer> nextLayer;
    private final float learningRate;

    public JoinedLayer(Optional<Layer> previousLayer, Optional<Layer> nextLayer, float learningRate) {
        this.previousLayer = previousLayer;
        this.nextLayer = nextLayer;
        this.learningRate = learningRate;
    }

    public JoinedLayer(Layer previousLayer, Layer nextLayer, float learningRate) {
        this(Optional.of(previousLayer), Optional.of(nextLayer), learningRate);
    }

    public final float[] forward(float[] input, boolean derive) {
        float[] previousLayerResult = previousLayer.get().forward(input, derive);
        return nextLayer.get().forward(previousLayerResult, derive);
    }

    public final float[] forward(float[] input) {
        return forward(input, false);
    }

    public final JoinedLayer backward(float[] input, float ideal) {
        Layer next = nextLayer.get();
        Layer previous = previousLayer.get();

        float[] forwarded = forward(input, true);
        float[] nextLayerDerivatives = next.getDerivatives();
        float[] nextLayerErrors = subtract(ideal, forwarded);

        float[] previousLayerErrors = new float[previous.getResults().length];

        for (int i = 0; i < next.getWeights().length; i++) {
            for (int j = 0; j < next.getWeights()[i].length; j++) {
                previousLayerErrors[j] += next.getWeights()[i][j] * nextLayerDerivatives[i] * nextLayerErrors[i];
            }
        }
        float[][] newPreviousLayerWeights = applyGradient(previous, input, previousLayerErrors);
        float[][] newNextLayerWeights = applyGradient(next, previous.getResults(), nextLayerErrors);
        return new JoinedLayer(new FullyConnectedLayer(newPreviousLayerWeights, previous.getTransferPair()), new FullyConnectedLayer(newNextLayerWeights, next.getTransferPair()), learningRate);
    }

    private float[][] applyGradient(Layer layer, float[] values, float[] errors) {
        float[][] newWeights = new float[layer.getWeights().length][];
        for (int i = 0; i < newWeights.length; i++) {
            newWeights[i] = new float[layer.getWeights()[i].length];
            for (int j = 0; j < newWeights[i].length; j++) {
                newWeights[i][j] = layer.getWeights()[i][j] + (values[j] * layer.getDerivatives()[i] * errors[i] * learningRate);
            }
        }
        return newWeights;
    }

    private float[] subtract(float number, float[] xs) {
        float[] results = new float[xs.length];
        for (int i = 0; i < xs.length; i++) {
            results[i] = number - xs[i];
        }
        return results;
    }

    @Override
    public final String toString() {
        return "(previousLayer=" + previousLayer.get() + ", nextLayer=" + nextLayer.get() + ")";
    }

}
