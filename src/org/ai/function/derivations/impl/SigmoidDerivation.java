package org.ai.function.derivations.impl;

import org.ai.function.derivations.Derivation;

public final class SigmoidDerivation implements Derivation {
    @Override
    public final float apply(float x) {
        return x * (1 - x);
    }

    @Override
    public final String toString() {
        return "Sigmoid";
    }
}
