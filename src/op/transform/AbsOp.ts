import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class AbsOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return Math.abs(a);
  }

}