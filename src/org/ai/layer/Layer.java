package org.ai.layer;

import org.ai.function.TransferFunctionPair;

import java.util.Random;

public abstract class Layer {
    protected final TransferFunctionPair transferPair;
    protected final float[][] weights;
    protected float[] results;

    private float[] derivatives;
    private final Random random;

    protected Layer(int size, int connections, long seed, TransferFunctionPair pair) {
        this.transferPair = pair;
        this.weights = new float[size][connections];
        this.random = getRandom(seed);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < connections; j++) {
                weights[i][j] = random.nextFloat();
            }
        }
    }

    protected Layer(float[][] weights, TransferFunctionPair pair) {
        this.weights = weights;
        this.transferPair = pair;
        this.random = null;
    }

    public abstract float[] forward(float[] input);

    public final float[] forward(float[] input, boolean derive) {
        float[] results = forward(input);
        if (derive) {
            this.results = results;
            this.derivatives = derive(results);
        }
        return results;
    }

    private float[] derive(float[] values) {
        float[] results = new float[values.length];
        for (int i = 0; i < results.length; i++) {
            results[i] = transferPair.getDerivation().apply(values[i]);
        }
        return results;
    }

    protected final float[] activate(float[] values) {
        float[] results = new float[values.length];
        for (int i = 0; i < results.length; i++) {
            results[i] = transferPair.getActivation().apply(values[i]);
        }
        return results;
    }


    private static Random getRandom(long seed) {
        return seed == -1 ? new Random() : new Random(seed);
    }

    public final float[] getDerivatives() {
        return derivatives;
    }

    public final float[] getResults() {
        return results;
    }

    public final float[][] getWeights() {
        return weights;
    }

    public final TransferFunctionPair getTransferPair() {
        return transferPair;
    }
}
