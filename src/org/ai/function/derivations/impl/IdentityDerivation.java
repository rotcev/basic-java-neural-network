package org.ai.function.derivations.impl;

import org.ai.function.derivations.Derivation;

public final class IdentityDerivation implements Derivation {
    @Override
    public final float apply(float x) {
        return 1;
    }

    @Override
    public final String toString() {
        return "Identity";
    }
}
