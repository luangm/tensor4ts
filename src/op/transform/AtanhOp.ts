import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class AtanhOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return Math.atanh(a);
  }

}