import { useState } from "react";
import KeyboardArrowDownIcon from '@material-ui/icons/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@material-ui/icons/KeyboardArrowUp';
import { TableSortLabel, TablePagination, IconButton, Collapse, Box, Typography, TableContainer, Paper, Table, TableHead, TableRow, TableCell, TableBody } from "@material-ui/core/";

function descendingComparator(a, b, orderBy) {
  if (b[orderBy] < a[orderBy]) {
    return -1;
  }
  if (b[orderBy] > a[orderBy]) {
    return 1;
  }
  return 0;
}

function getComparator(order, orderBy) {
  return order === 'desc'
    ? (a, b) => descendingComparator(a, b, orderBy)
    : (a, b) => -descendingComparator(a, b, orderBy);
}

function stableSort(array, comparator) {
  const stabilizedThis = array.map((el, index) => [el, index]);
  stabilizedThis.sort((a, b) => {
    const order = comparator(a[0], b[0]);
    if (order !== 0) return order;
    return a[1] - b[1];
  });
  return stabilizedThis.map((el) => el[0]);
}

export default function ReportTable({ scores }) {

  const [page, setPage] = useState(0);
  const [rowsPerPage] = useState(5);
  const [order, setOrder] = useState('desc');
  const [orderBy, setOrderBy] = useState('labels');

  const handleRequestSort = (event, property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };
  const sortable = [
    { id: 'labels', numeric: false, label: 'Number of Labels' },
    { id: 'accuracy', numeric: true, label: 'Accuracy' }
  ];

  return (
    <TableContainer component={Paper}>
      <Table aria-label="collapsible table">
        <TableHead>
          <TableRow>
            <TableCell />
            {sortable.map((col) => (
              <TableCell
                key={col.id}
                align={col.numeric ? 'right' : 'left'}
                sortDirection={orderBy === col.id ? order : false}>
                <TableSortLabel
                  active={orderBy === col.id}
                  direction={orderBy === col.id ? order : 'asc'}
                  onClick={(event) => handleRequestSort(event, col.id)}>
                  {col.label}
                </TableSortLabel>
              </TableCell>
            ))}
            <TableCell align="right">Macro F1</TableCell>
            <TableCell align="right">Weighted F1</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {stableSort(scores, getComparator(order, orderBy))
            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row) => (
              <Row key={row.labels} row={row} />
            ))}
        </TableBody>
      </Table>
      <TablePagination
        rowsPerPageOptions={[5]}
        component="div"
        count={scores.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onChangePage={(e, page) => setPage(page)}
      />
    </TableContainer>
  );
}

function Row(props) {
  const { row } = props;
  const [open, setOpen] = useState(false);

  return (
    <>
      <TableRow >
        <TableCell>
          <IconButton aria-label="expand row" size="small" onClick={() => setOpen(!open)}>
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell component="th" scope="row">
          {row.labels}
        </TableCell>
        <TableCell align="right">{(row.accuracy * 100).toFixed(2)}%</TableCell>
        <TableCell align="right">{(row['macro avg']['f1-score'] * 100).toFixed(2)}%</TableCell>
        <TableCell align="right">{(row['weighted avg']['f1-score'] * 100).toFixed(2)}%</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box marginBottom={2}>
              <Typography variant="h6" gutterBottom component="div">
                Detailed Classification Report
              </Typography>
              <Table size="small" aria-label="summary">
                <TableHead>
                  <TableRow>
                    <TableCell />
                    <TableCell align="center">Precision</TableCell>
                    <TableCell align="center">Recall</TableCell>
                    <TableCell align="center">F1-score</TableCell>
                    <TableCell align="center">Support</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody style={{ textTransform: "capitalize" }}>
                  {Object.entries(row).map((key) =>
                    <TableRow key={key[0]}>
                      {(() => {
                        let i = 0;
                        switch (key[0]) {
                          case "accuracy": return (
                            [<TableCell key={0}>{key[0]}</TableCell>,
                            <TableCell key={1}></TableCell>,
                            <TableCell key={2}></TableCell>,
                            <TableCell align="center" key={3}>{key[1].toFixed(2)}</TableCell>,
                            <TableCell key={4}></TableCell>]);
                          case "labels": return null
                          default: return (
                            [<TableCell key={key[0]}>{key[0]}</TableCell>,
                            Object.entries(key[1]).map((val) => (
                              <TableCell align="center" key={i++}>{val[0] === "support" ? val[1] : val[1].toFixed(2)}</TableCell>
                            ))]);
                        }
                      })()}
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}