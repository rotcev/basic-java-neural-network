package org.ai.function.activations.impl;

import org.ai.function.activations.Activation;

public final class RectifiedLinearUnitActivation implements Activation {
    @Override
    public final float apply(float x) {
        return Math.max(0, x);
    }

    @Override
    public final String toString() {
        return "ReLU";
    }
}
