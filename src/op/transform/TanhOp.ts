import TransformOp from "./TransformOp";

export default class TanhOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.tanh(a);
  }

}