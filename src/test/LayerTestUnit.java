package test;

import org.ai.function.activations.impl.BinaryActivation;
import org.ai.function.activations.impl.RectifiedLinearUnitActivation;
import org.ai.function.activations.impl.SigmoidActivation;
import org.ai.function.derivations.impl.IdentityDerivation;
import org.ai.function.derivations.impl.RectifiedLinearUnitDerivation;
import org.ai.function.derivations.impl.SigmoidDerivation;
import org.ai.layer.impl.FullyConnectedLayer;
import org.ai.layer.JoinedLayer;
import org.ai.layer.JoinedLayerList;
import org.ai.layer.Layer;
import org.junit.Assert;
import org.junit.Test;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

public class LayerTestUnit {

    @Test
    public void testLayerWeightTopologies() {
        Optional<Layer> hiddenLayer = FullyConnectedLayer.of(SigmoidActivation.class, SigmoidDerivation.class, 3, 2, 1);

        hiddenLayer.ifPresent(l -> {
            assertEquals(3, l.getWeights().length);

            for (int i = 0; i < l.getWeights().length; i++) {
                assertEquals(2, l.getWeights()[i].length);
            }
        });
    }

    @Test
    public void testForward() {
        Optional<Layer> hiddenLayer = FullyConnectedLayer.of(SigmoidActivation.class, SigmoidDerivation.class, 3, 2, 1);//3 neurons, 2 connections = 6 synapses
        Optional<Layer> outputLayer = FullyConnectedLayer.of(SigmoidActivation.class, SigmoidDerivation.class, 1, 3, 334345345);//1 neuron, 3 connections = 3 synapses

        Optional<Layer> singleInputLayer = FullyConnectedLayer.of(SigmoidActivation.class, SigmoidDerivation.class, 1, 1, 3423432);//1 neuron, 1 connection = 1 synapse
        Optional<Layer> singleInputLayer2 = FullyConnectedLayer.of(SigmoidActivation.class, SigmoidDerivation.class, 1, 1, 232312358);//1 neuron, 1 connection = 1 synapse

        JoinedLayer joinedLayer = new JoinedLayer(hiddenLayer, outputLayer, 0.1f);
        JoinedLayer secondJoinedLayer = new JoinedLayer(singleInputLayer, singleInputLayer2, 0.1f);

        float[] input = new float[]{1, 1};
        JoinedLayerList joinedLayerList = new JoinedLayerList(joinedLayer, secondJoinedLayer);

        System.out.println(Arrays.deepToString(joinedLayerList.forward(input)));
        System.out.println(Arrays.toString(singleInputLayer2.get().forward(singleInputLayer.get().forward(outputLayer.get().forward(hiddenLayer.get().forward(input))))));

        assertEquals(Arrays.toString(joinedLayerList.forward(input)[0]), Arrays.toString(outputLayer.get().forward(hiddenLayer.get().forward(input))));
        assertEquals(Arrays.toString(joinedLayerList.forward(input)[1]), Arrays.toString(singleInputLayer2.get().forward(singleInputLayer.get().forward(outputLayer.get().forward(hiddenLayer.get().forward(input))))));
    }

    @Test
    public void testBackwardByXOR() {
        Optional<Layer> hiddenLayer = FullyConnectedLayer.of(BinaryActivation.class, IdentityDerivation.class, 3, 2, 1);//3 neurons, 2 connections = 6 synapses
        Optional<Layer> outputLayer = FullyConnectedLayer.of(BinaryActivation.class, IdentityDerivation.class, 1, 3, 334345345);//1 neuron, 3 connections = 3 synapses

        JoinedLayer joinedLayer = new JoinedLayer(hiddenLayer, outputLayer, 0.1f);
        float[][] input = new float[][]{{1, 1}, {0, 1}, {0, 0}, {1, 0}};
        float[] ideal = new float[]{0, 1, 0, 1};

        System.out.println("Prior to training:");
        for (int i = 0; i < input.length; i++) {
            float[] result = joinedLayer.forward(input[i]);
            System.out.println("\t"+Arrays.toString(input[i]) + " is thought to be " + Arrays.toString(result)+", should be "+ideal[i]);
        }

        List<Integer> indexes = IntStream.range(0, input.length).boxed().collect(Collectors.toList());
        for (int i = 0; i < 100000; i++) {
            Collections.shuffle(indexes);

            for (int index : indexes) {
                joinedLayer = joinedLayer.backward(input[index], ideal[index]);
            }
        }


        System.out.println("After back propagation:");
        for (int i = 0; i < input.length; i++) {
            float[] result = joinedLayer.forward(input[i]);
            System.out.println("\t"+Arrays.toString(input[i]) + " is thought to be " + Arrays.toString(result)+", should be "+ideal[i]);
            Assert.assertEquals(result[0], ideal[i], 0.05);
        }
    }

    @Test
    public void testOneHotVectorInput() {
        Optional<Layer> hiddenLayer = FullyConnectedLayer.of(RectifiedLinearUnitActivation.class, RectifiedLinearUnitDerivation.class, 10, 10, 1);//10 neurons, 10 connections = 100 synapses
        Optional<Layer> outputLayer = FullyConnectedLayer.of(RectifiedLinearUnitActivation.class, RectifiedLinearUnitDerivation.class, 1, 10, 334345345);//1 neuron, 10 connections = 10 synapses

        JoinedLayer joinedLayer = new JoinedLayer(hiddenLayer, outputLayer, 0.1f);
        float[][] input = new float[][]{
                {1,0,0,0,0,0,0,0,0,0}, {0,1,0,0,0,0,0,0,0,0},
                {0,0,1,0,0,0,0,0,0,0}, {0,0,0,1,0,0,0,0,0,0},
                {0,0,0,0,1,0,0,0,0,0}, {0,0,0,0,0,1,0,0,0,0},
                {0,0,0,0,0,0,1,0,0,0}, {0,0,0,0,0,0,0,1,0,0},
                {0,0,0,0,0,0,0,0,1,0}, {0,0,0,0,0,0,0,0,0,1}
        };
        float[] ideal = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        System.out.println("Prior to training:");
        for (int i = 0; i < input.length; i++) {
            float[] result = joinedLayer.forward(input[i]);
            System.out.println("\t"+Arrays.toString(input[i]) + " is thought to be " + Arrays.toString(result)+", should be "+ideal[i]);
        }

        List<Integer> indexes = IntStream.range(0, input.length).boxed().collect(Collectors.toList());
        for (int i = 0; i < 200000; i++) {
            Collections.shuffle(indexes);

            for (int index : indexes) {
                joinedLayer = joinedLayer.backward(input[index], ideal[index]);
            }
        }

        System.out.println("After back propagation:");
        for (int i = 0; i < input.length; i++) {
            float[] result = joinedLayer.forward(input[i]);
            System.out.println("\t"+Arrays.toString(input[i]) + " is thought to be " + Arrays.toString(result)+", should be "+ideal[i]);
//            System.out.println(result[i]+", "+ideal[i]);
            Assert.assertEquals(result[0], ideal[i], 0.05);
//            Assert.assertEquals(Arrays.toString(result), "["+ideal[i]+"]");
        }
    }
}