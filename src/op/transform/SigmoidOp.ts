import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class SigmoidOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return 1 / (1 + Math.exp(-a));
  }

}