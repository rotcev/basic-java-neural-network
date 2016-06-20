package org.ai.function;

import org.ai.function.activations.Activation;
import org.ai.function.derivations.Derivation;

public final class TransferFunctionPair<A extends Activation, D extends Derivation> {

    private final A activation;
    private final D derivation;

    public TransferFunctionPair(A activation, D derivation) {
        this.activation = activation;
        this.derivation = derivation;
    }

    public final A getActivation() {
        return activation;
    }

    public final D getDerivation() {
        return derivation;
    }

    @Override
    public final String toString() {
        return "activation="+activation+", derivation="+ derivation;
    }
}
