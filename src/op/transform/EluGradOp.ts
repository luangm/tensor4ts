import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class EluGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return a >= 0 ? 1 : Math.exp(a);
  }

}