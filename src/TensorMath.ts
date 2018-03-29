import Executor from "./executor/Executor";
import {Col2ImOptions, default as Col2ImOp} from "./op/cnn/Col2ImOp";
import {default as Im2ColOp, Im2ColOptions} from "./op/cnn/Im2ColOp";
import ArangeOp from "./op/creation/ArangeOp";
import LinspaceOp from "./op/creation/LinspaceOp";
import ArgMaxOp from "./op/index/ArgMaxOp";
import ArgMinOp from "./op/index/ArgMinOp";
import AddOp from "./op/pairwise/AddOp";
import DivideOp from "./op/pairwise/DivideOp";
import MaxOp from "./op/pairwise/MaxOp";
import MinOp from "./op/pairwise/MinOp";
import ModOp from "./op/pairwise/ModOp";
import MultiplyOp from "./op/pairwise/MultiplyOp";
import SubtractOp from "./op/pairwise/SubtractOp";
import ReduceMaxOp from "./op/reduction/ReduceMaxOp";
import ReduceMeanOp from "./op/reduction/ReduceMeanOp";
import ReduceMinOp from "./op/reduction/ReduceMinOp";
import ReduceProdOp from "./op/reduction/ReduceProdOp";
import ReduceSumOp from "./op/reduction/ReduceSumOp";
import MatMulOp from "./op/special/MatMulOp";
import AbsOp from "./op/transform/AbsOp";
import AcoshOp from "./op/transform/AcoshOp";
import AcosOp from "./op/transform/AcosOp";
import AsinhOp from "./op/transform/AsinhOp";
import AsinOp from "./op/transform/AsinOp";
import AtanhOp from "./op/transform/AtanhOp";
import AtanOp from "./op/transform/AtanOp";
import CeilOp from "./op/transform/CeilOp";
import CoshOp from "./op/transform/CoshOp";
import CosOp from "./op/transform/CosOp";
import EluOp from "./op/transform/EluOp";
import Expm1Op from "./op/transform/Expm1Op";
import ExpOp from "./op/transform/ExpOp";
import FloorOp from "./op/transform/FloorOp";
import Log1pOp from "./op/transform/Log1pOp";
import LogOp from "./op/transform/LogOp";
import NegateOp from "./op/transform/NegateOp";
import PowerOp from "./op/transform/PowerOp";
import RandomOp from "./op/transform/RandomOp";
import ReciprocalOp from "./op/transform/ReciprocalOp";
import ReluOp from "./op/transform/ReluOp";
import RoundOp from "./op/transform/RoundOp";
import RsqrtOp from "./op/transform/RsqrtOp";
import SigmoidGradOp from "./op/transform/SigmoidGradOp";
import SigmoidOp from "./op/transform/SigmoidOp";
import SignOp from "./op/transform/SignOp";
import SinhOp from "./op/transform/SinhOp";
import SinOp from "./op/transform/SinOp";
import SoftplusOp from "./op/transform/SoftplusOp";
import SetOp from "./op/transform/special/SetOp";
import SqrtGradOp from "./op/transform/SqrtGradOp";
import SqrtOp from "./op/transform/SqrtOp";
import SquareOp from "./op/transform/SquareOp";
import StepOp from "./op/transform/StepOp";
import TanGradOp from "./op/transform/TanGradOp";
import TanhOp from "./op/transform/TanhOp";
import TanOp from "./op/transform/TanOp";
import Tensor from "./Tensor";
import ShapeUtils from "./utils/ShapeUtils";

export default class TensorMath {

  static abs(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AbsOp(base, null, result));
    return result;
  }

  static acos(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AcosOp(base, null, result));
    return result;
  }

  static acosh(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AcoshOp(base, null, result));
    return result;
  }

  static add(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new AddOp(left, right, result));
    return result;
  }

  // TODO
  static addN(items: Tensor[]): Tensor {
    return null;
  }

  static arange(base: Tensor, stop: number, start: number = 0, step: number = 1, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new ArangeOp(base, null, result, stop, start, step));
    return result;
  }

  static argMax(base: Tensor, dim: number = -1, keepDims: boolean = false): Tensor {
    let resultShape = ShapeUtils.reduceShape(base.shape, dim, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ArgMaxOp(base, null, result, dim));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dim, false);
    return result.reshape(reducedShape);
  }

  static argMin(base: Tensor, dim: number = -1, keepDims: boolean = false): Tensor {
    let resultShape = ShapeUtils.reduceShape(base.shape, dim, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ArgMinOp(base, null, result, dim));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dim, false);
    return result.reshape(reducedShape);
  }

  // TODO
  static argSet(source: Tensor, args: number, shape: number, dim: number): Tensor {
    // let result = new Tensor({shape: shape});
    // let op = new IndexSetOp(source, args, result);
    // Executor.execAtDim(op, dim);
    // return result;
    return null;
  }

  static asin(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AsinOp(base, null, result));
    return result;
  }

  static asinh(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AsinhOp(base, null, result));
    return result;
  }

  static atan(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AtanOp(base, null, result));
    return result;
  }

  // static conv2dImageGrad(image, kernel, grad) {
  //   let numKernels = kernel.shape[0];
  //
  //   let gradReshape = grad.reshape([numKernels, grad.length / numKernels]);
  //   let kReshape = kernel.reshape([numKernels, kernel.length / numKernels]);
  //   let col = TensorMath.matmul(kReshape, gradReshape, true, false);
  //
  //   return TensorUtils.col2im(col, image, kernel).reshape(image.shape);
  // }
  //
  // static conv2dKernelGrad(image, kernel, grad) {
  //   let numKernels = kernel.shape[0];
  //   let xCol = TensorUtils.im2col(image, kernel);
  //   let gradReshape = grad.reshape([numKernels, grad.length / numKernels]);
  //   return TensorMath.matmul(gradReshape, xCol, false, true).reshape(kernel.shape);
  // }

  static atanh(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new AtanhOp(base, null, result));
    return result;
  }

  static ceil(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new CeilOp(base, null, result));
    return result;
  }

  static col2im(base: Tensor, options: Col2ImOptions): Tensor {
    let result = Tensor.zeros([options.imageNum, options.imageChannel, options.imageHeight, options.imageWidth]);
    Executor.exec(new Col2ImOp(base, null, result, options));
    return result;
  }

  // TODO
  static conv2d(image: Tensor, kernel: Tensor): Tensor {
    return null;
    // let xCol = TensorUtils.im2col(image, kernel.shape);
    //
    // let numImages = image.shape[0];
    // let channels = image.shape[1];
    // let height = image.shape[2]; // rows
    // let width = image.shape[3]; // cols
    //
    // let numKernels = kernel.shape[0];
    // let kernelChannels = kernel.shape[1];
    // let kernelHeight = kernel.shape[2]; // rows
    // let kernelWidth = kernel.shape[3]; // cols
    //
    // let outputHeight = TensorUtils.computeConv2dOutSize(height, kernelHeight);
    // let outputWidth = TensorUtils.computeConv2dOutSize(width, kernelWidth);
    //
    // let kCol = kernel.reshape([numKernels, kernelChannels * kernelWidth * kernelHeight]);
    // let result = TensorMath.matmul(kCol, xCol);
    // let reshaped = result.reshape([numKernels, numImages, outputHeight, outputWidth]);
    // let transposed = reshaped.transpose([1, 0, 2, 3]);
    // return transposed;
  }

  static cos(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new CosOp(base, null, result));
    return result;
  }

  static cosh(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new CoshOp(base, null, result));
    return result;
  }

  static divide(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new DivideOp(left, right, result));
    return result;
  }

  static elu(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new EluOp(base, null, result));
    return result;
  }

  static exp(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new ExpOp(base, null, result));
    return result;
  }

  // static logSumExp(base, dim = -1) {
  //   if (dim < 0) {
  //     dim += base.rank;
  //   }
  //   let max = TensorMath.reduceMax(base, dim, true);
  //   let subtract = TensorMath.subtract(base, max);
  //   let exp = TensorMath.exp(subtract);
  //   let sum = TensorMath.reduceSum(exp, dim, true);
  //   let log = TensorMath.log(sum);
  //   return TensorMath.add(log, max);
  // }

  static expm1(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new Expm1Op(base, null, result));
    return result;
  }

  static fill(base: Tensor, scalar: number, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SetOp(base, null, result, scalar));
    return result;
  }

  static floor(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new FloorOp(base, null, result));
    return result;
  }

  // static maxPool(image, kernelShape, strideWidth, strideHeight) {
  //
  //   let numImages = image.shape[0];
  //   let channels = image.shape[1];
  //   let height = image.shape[2]; // rows
  //   let width = image.shape[3]; // cols
  //
  //   let numKernels = kernelShape[0];
  //   let kernelChannels = kernelShape[1];
  //   let kernelHeight = kernelShape[2]; // rows
  //   let kernelWidth = kernelShape[3]; // cols
  //
  //   let outputHeight = TensorUtils.computeConv2dOutSize(height, kernelHeight, 0, strideHeight);
  //   let outputWidth = TensorUtils.computeConv2dOutSize(width, kernelWidth, 0, strideWidth);
  //
  //   let xCol = TensorUtils.im2col(image, kernelShape, {strideWidth, strideHeight});
  //   let max = TensorMath.reduceMax(xCol, 0);
  //   let result = max.reshape([numImages, channels, outputHeight, outputWidth]);
  //   return result;
  // }
  //
  // static maxPoolGrad(image, kernel, grad, {strideWidth, strideHeight}) {
  //   let xCol = TensorUtils.im2col(image, kernel.shape, {strideWidth, strideHeight});
  //   let argmax = TensorMath.argMax(xCol, 0);
  //   let gradReshape = grad.reshape([1, grad.length]);
  //   let set = TensorMath.argSet(gradReshape, argmax, xCol.shape, 0);
  //   let result = TensorUtils.col2im(set, image, kernel, {strideWidth, strideHeight}).reshape(image.shape);
  //   return result;
  // }

  static im2col(base: Tensor, options: Im2ColOptions): Tensor {
    let imageNum = base.shape[0];
    let imageChannel = base.shape[1];
    let imageHeight = base.shape[2]; // rows
    let imageWidth = base.shape[3]; // cols

    let kernelNum = options.kernelNum;
    let kernelChannel = options.kernelChannel;
    let kernelHeight = options.kernelHeight; // rows
    let kernelWidth = options.kernelWidth; // cols

    let padHeight = options.padHeight;
    let padWidth = options.padWidth;
    let strideHeight = options.strideHeight;
    let strideWidth = options.strideWidth;

    let outputHeight = ShapeUtils.computeConvOutSize(imageHeight, kernelHeight, padHeight, strideHeight);
    let outputWidth = ShapeUtils.computeConvOutSize(imageWidth, kernelWidth, padWidth, strideWidth);
    let resultHeight = kernelChannel * kernelHeight * kernelWidth;
    let resultWidth = imageNum * outputHeight * outputWidth;

    let result = Tensor.zeros([resultHeight, resultWidth]);

    Executor.exec(new Im2ColOp(base, null, result, options));

    return result;
  }

  static linspace(base: Tensor, start: number, stop: number = 0, num: number, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new LinspaceOp(base, null, result, start, stop, num));
    return result;
  }

  static log(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new LogOp(base, null, result));
    return result;
  }

  static log1p(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new Log1pOp(base, null, result));
    return result;
  }

  static matmul(left: Tensor, right: Tensor, transposeLeft = false, transposeRight = false, result?: Tensor): Tensor {
    if (left.rank !== 2 || right.rank !== 2) {
      throw new Error('Invalid Operation, Rank of left and right must be 2');
    }

    let shape = [0, 0];
    shape[0] = transposeLeft ? left.shape[1] : left.shape[0];
    shape[1] = transposeRight ? right.shape[0] : right.shape[1];
    result = result || Tensor.zeros(shape);
    Executor.exec(new MatMulOp(left, right, result, transposeLeft, transposeRight));
    return result;
  }

  static max(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new MaxOp(left, right, result));
    return result;
  }

  static min(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new MinOp(left, right, result));
    return result;
  }

  static mod(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new ModOp(left, right, result));
    return result;
  }

  static multiply(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new MultiplyOp(left, right, result));
    return result;
  }

  static negate(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new NegateOp(base, null, result));
    return result;
  }

  static pow(base: Tensor, power: number = 1, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new PowerOp(base, null, result, power));
    return result;
  }

  static rand(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new RandomOp(base, null, result));
    return result;
  }

  static reciprocal(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new ReciprocalOp(base, null, result));
    return result;
  }

  static reduceMax(base: Tensor, dims: number | number[] = -1, keepDims: boolean = false): Tensor {
    let reducedDims = ShapeUtils.getReducedDims(base.shape, dims);
    let resultShape = ShapeUtils.reduceShape(base.shape, dims, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ReduceMaxOp(base, null, result, reducedDims));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dims, false);
    return result.reshape(reducedShape);
  }

  static reduceMean(base: Tensor, dims: number | number[] = -1, keepDims: boolean = false): Tensor {
    let reducedDims = ShapeUtils.getReducedDims(base.shape, dims);
    let resultShape = ShapeUtils.reduceShape(base.shape, dims, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ReduceMeanOp(base, null, result, reducedDims));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dims, false);
    return result.reshape(reducedShape);
  }

  static reduceMin(base: Tensor, dims: number | number[] = -1, keepDims: boolean = false): Tensor {
    let reducedDims = ShapeUtils.getReducedDims(base.shape, dims);
    let resultShape = ShapeUtils.reduceShape(base.shape, dims, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ReduceMinOp(base, null, result, reducedDims));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dims, false);
    return result.reshape(reducedShape);
  }

  static reduceProd(base: Tensor, dims: number | number[] = -1, keepDims: boolean = false): Tensor {
    let reducedDims = ShapeUtils.getReducedDims(base.shape, dims);
    let resultShape = ShapeUtils.reduceShape(base.shape, dims, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ReduceProdOp(base, null, result, reducedDims));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dims, false);
    return result.reshape(reducedShape);
  }

  static reduceSum(base: Tensor, dims: number | number[] = -1, keepDims: boolean = false): Tensor {
    let reducedDims = ShapeUtils.getReducedDims(base.shape, dims);
    let resultShape = ShapeUtils.reduceShape(base.shape, dims, true);
    let result = Tensor.zeros(resultShape);
    Executor.exec(new ReduceSumOp(base, null, result, reducedDims));
    if (keepDims) {
      return result;
    }
    let reducedShape = ShapeUtils.reduceShape(base.shape, dims, false);
    return result.reshape(reducedShape);
  }

  static relu(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new ReluOp(base, null, result));
    return result;
  }

  static round(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new RoundOp(base, null, result));
    return result;
  }

  static rsqrt(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new RsqrtOp(base, null, result));
    return result;
  }

  static sigmoid(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SigmoidOp(base, null, result));
    return result;
  }

  static sigmoidGrad(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SigmoidGradOp(base, null, result));
    return result;
  }

  static sign(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SignOp(base, null, result));
    return result;
  }

  // static softmaxCrossEntropyGrad(labels, logits) {
  //   let softmax = TensorMath(logits);
  //   return TensorMath.subtract(softmax, labels);
  // }
  //
  // static softmaxCrossEntropyWithLogits(labels, logits, dim = -1) {
  //   if (dim < 0) {
  //     dim += logits.rank;
  //   }
  //   let logSumExp = TensorMath.logSumExp(logits);
  //   let sub = TensorMath.subtract(logits, logSumExp);
  //   let mul = TensorMath.multiply(labels, sub);
  //   let sum = TensorMath.reduceSum(mul, dim);
  //   return TensorMath.negate(sum);
  // }

  // /**
  //  * Normally a softmax derivative is a Jacobian Matrix.
  //  * To get a total derivative, sum up all the partials.
  //  *
  //  * Assume shape of base = [batch, elements]
  //  */
  // static softmaxGrad(base, grad) {
  //   let softmax = TensorMath.softmax(base); // default dim = -1
  //   let mul = TensorMath.multiply(grad, softmax);
  //   let sum = TensorMath.reduceSum(mul, 1); // reduce on last dim
  //   let subtract = TensorMath.subtract(grad, sum); // Sum will broadcast
  //   return TensorMath.multiply(subtract, softmax);
  // }

  static sin(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SinOp(base, null, result));
    return result;
  }

  static sinh(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SinhOp(base, null, result));
    return result;
  }

  static softmax(base: Tensor, dim: number = -1): Tensor {
    if (dim < 0) {
      dim += base.rank;
    }
    let max = TensorMath.reduceMax(base, dim, true);
    let subtract = TensorMath.subtract(base, max);
    let exp = TensorMath.exp(subtract);
    let sum = TensorMath.reduceSum(exp, dim, true);
    return TensorMath.divide(exp, sum);
  }

  static softplus(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SoftplusOp(base, null, result));
    return result;
  }

  static sqrt(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SqrtOp(base, null, result));
    return result;
  }

  static sqrtGrad(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SqrtGradOp(base, null, result));
    return result;
  }

  static square(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new SquareOp(base, null, result));
    return result;
  }

  static step(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new StepOp(base, null, result));
    return result;
  }

  static subtract(left: Tensor, right: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
    Executor.exec(new SubtractOp(left, right, result));
    return result;
  }

  // static sumSquaredError(label, prediction) {
  //   let sub = TensorMath.subtract(label, prediction);
  //   let sqr = TensorMath.square(sub);
  //   return TensorMath.reduceSum(sqr, -1);
  // }

  static tan(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new TanOp(base, null, result));
    return result;
  }

  static tanGrad(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new TanGradOp(base, null, result));
    return result;
  }

  static tanh(base: Tensor, result?: Tensor): Tensor {
    result = result || Tensor.zeros(base.shape);
    Executor.exec(new TanhOp(base, null, result));
    return result;
  }

  // TODO: This is a Hack. TB Fixed
  static tile(base: Tensor, repeats: number[]): Tensor {
    // let shape = base.shape.slice();
    // for (let i = 0; i < shape.length; i++) {
    //   shape[i] *= repeats[i];
    // }
    // let result = new Tensor({shape});
    // Executor.exec(new SetOp(result, null, result, {scalar: base.data[0]}));
    // return result;
    return null;
  }
}