package org.ai.layer;

import java.util.Arrays;
import java.util.List;

public final class JoinedLayerList {
    private final List<JoinedLayer> joinedLayers;

    public JoinedLayerList(List<JoinedLayer> joinedLayers) {
        this.joinedLayers = joinedLayers;
    }

    public JoinedLayerList(JoinedLayer... joinedLayers) {
        this(Arrays.asList(joinedLayers));
    }

    public final float[][] forward(float[] input) {
        float[] toForward = input;
        float[][] results = new float[joinedLayers.size()][];

        for (int i = 0; i < results.length; i++) {
            results[i] = toForward = joinedLayers.get(i).forward(toForward);
        }
        return results;
    }
}
