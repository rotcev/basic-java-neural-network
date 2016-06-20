package org.ai.function.activations.impl;

import org.ai.function.activations.Activation;

public final class SigmoidActivation implements Activation {
    @Override
    public final float apply(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    @Override
    public String toString() {
        return "Sigmoid";
    }
}
