package org.ai.function.derivations.impl;

import org.ai.function.derivations.Derivation;

public final class RectifiedLinearUnitDerivation implements Derivation {
    @Override
    public final float apply(float x) {
        return x > 0 ? 1 : 0;
    }

    @Override
    public final String toString() {
        return "ReLU";
    }
}
