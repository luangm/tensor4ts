import TransformOp from "./TransformOp";

export default class NegateOp extends TransformOp {

  body(a: number, b?: number): number {
    return -a;
  }

}