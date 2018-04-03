import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class SqrtGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return 0.5 / Math.sqrt(a);
  }

}