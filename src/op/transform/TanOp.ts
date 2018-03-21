import TransformOp from "./TransformOp";

export default class TanOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.tan(a);
  }

}