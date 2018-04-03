import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class SigmoidGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    let sigmoid = 1 / (1 + Math.exp(-a));
    return sigmoid * (1.0 - sigmoid);
  }

}