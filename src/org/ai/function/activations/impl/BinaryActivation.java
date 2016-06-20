package org.ai.function.activations.impl;

import org.ai.function.activations.Activation;

public final class BinaryActivation implements Activation {
    @Override
    public final float apply(float x) {
        return x >= 0 ? 1 : 0;
    }

    @Override
    public final String toString() {
        return "Binary";
    }
}
