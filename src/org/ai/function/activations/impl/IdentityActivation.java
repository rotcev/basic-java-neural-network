package org.ai.function.activations.impl;

import org.ai.function.activations.Activation;

public final class IdentityActivation implements Activation {
    @Override
    public final float apply(float x) {
        return x;
    }

    @Override
    public final String toString() {
        return "Identity";
    }
}
